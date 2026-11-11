"""Process coordination for the static Verl V1 separate_async integration."""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from inspect import signature
from pathlib import Path

from elastic_rollout.client import ControlClient
from elastic_rollout.launcher.owned_exec import owned_command

from .launch_spec import LaunchSpec, validate_verl_config

logger = logging.getLogger(__name__)


@contextmanager
def ray_cluster(address):
    """Inspect the existing cluster without retaining a driver or replacing signals."""
    if address.startswith("ray://") or address == "local":
        raise ValueError("MVP-2B requires an existing native Ray cluster address")
    import ray

    if ray.is_initialized():
        raise RuntimeError("launcher must resolve Ray before another driver initializes it")
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        context = ray.init(address=address)
        yield context, ray
    finally:
        try:
            ray.shutdown()
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def resolve_ray_cluster(spec: LaunchSpec, core_config: dict) -> str:
    """Verify rollout slots and resolve the GCS address before spawning children."""
    with ray_cluster(spec.ray_address) as (context, ray):
        address = context.address_info.get("gcs_address")
        if not address or address in ("auto", "local"):
            raise RuntimeError("Ray did not return a concrete GCS address")
        nodes = {node["NodeID"]: node for node in ray.nodes() if node.get("Alive")}
        if core_config.get("resource_backend") == "ray_pg":
            return address  # Capacity may arrive after platform provisioning.
        slots_by_node: dict[str, int] = {}
        for slot in core_config["slots"]:
            node_id = slot["node_id"]
            if node_id not in nodes:
                raise ValueError(f"rollout slot refers to missing Ray node {node_id}")
            slots_by_node[node_id] = slots_by_node.get(node_id, 0) + slot["num_devices"]
        for node_id, count in slots_by_node.items():
            resource = "GPU" if spec.accelerator == "gpu" else "NPU"
            available = nodes[node_id].get("Resources", {}).get(resource, 0)
            if count > available:
                raise ValueError(
                    f"rollout slots need {count} {resource}s on {node_id}, Ray has {available}"
                )
        logger.info("resolved Ray cluster GCS %s with %d live nodes", address, len(nodes))
        return address


def validate_bootstrap(status: dict, spec: LaunchSpec, core_config: dict) -> None:
    """Check identity once; Core owns the changing topology predicate."""
    deployment = status.get("spec", {})
    replica = deployment.get("replica", {})
    if deployment.get("deployment_id") != spec.deployment_id:
        raise RuntimeError("Core deployment identity differs from LaunchSpec")
    if replica.get("model") != spec.model_path or replica.get("weight_sync") != spec.backend:
        raise RuntimeError("Core bootstrap model or transport differs from LaunchSpec")
    if replica.get("policy_version") != spec.initial_version:
        raise RuntimeError("Core bootstrap replica version differs from LaunchSpec")
    bootstrap = status.get("bootstrap_snapshot", {})
    if (
        bootstrap.get("model_path") != spec.model_path
        or bootstrap.get("version") != spec.initial_version
    ):
        raise RuntimeError("Core bootstrap snapshot differs from LaunchSpec")
    if status.get("desired_replicas") != core_config["deployments"][0]["desired_replicas"]:
        raise RuntimeError("Core desired replica count differs from Core launch config")
    if status.get("serving_policy_version") != spec.initial_version:
        raise RuntimeError("Core serving version differs from bootstrap")


def _stop_child(child: subprocess.Popen | None, timeout: float) -> int | None:
    if child is None:
        return None
    code = child.poll()
    if code is not None:
        return code
    logger.info("requesting child shutdown pid=%s timeout=%.1fs", child.pid, timeout)
    try:
        child.terminate()
    except ProcessLookupError:
        return child.wait(timeout=timeout)
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("child %s did not exit within %.1fs; forcing exit", child.pid, timeout)
        child.kill()
        child.wait()
        return None


def _wait_ready(
    client: ControlClient, spec: LaunchSpec, core_config: dict, core: subprocess.Popen, stop
) -> dict:
    deadline = time.monotonic() + spec.startup_timeout_s
    last_error = "Core control API unavailable"
    validated = False
    while time.monotonic() < deadline:
        if stop[0]:
            raise InterruptedError("launch interrupted before serving readiness")
        if core.poll() is not None:
            raise RuntimeError(f"Core exited before serving readiness (exit {core.returncode})")
        try:
            status = client.status()
            if "base_failed" in status.get("route_pause_reasons", ()):
                raise RuntimeError(
                    f"Core base replica failed before trainer start: {status.get('last_error')}"
                )
            if not validated:
                validate_bootstrap(status, spec, core_config)
                validated = True
            if status.get("serving_ready_for_external_update"):
                if (
                    status.get("desired_replicas")
                    != core_config["deployments"][0]["desired_replicas"]
                ):
                    raise RuntimeError("Core desired topology changed before trainer start")
                if status.get("serving_policy_version") != spec.initial_version:
                    raise RuntimeError("Core serving version changed before trainer start")
                return status
            last_error = (
                f"ready={status.get('ready_replicas')}, desired={status.get('desired_replicas')}, "
                f"stable={status.get('serving_ready_for_external_update')}, "
                f"error={status.get('last_error')}"
            )
        except (ConnectionError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"Core did not become serving_ready: {last_error}")


def run_launch(
    path: Path | LaunchSpec,
    *,
    report_path: Path | None = None,
    trainer_config=None,
) -> int:
    spec = path if isinstance(path, LaunchSpec) else LaunchSpec.from_file(path)
    core_config = spec.core_config()
    # Resolve and verify every input before spawning either child.
    ray_address = (
        resolve_ray_cluster(spec, core_config) if trainer_config is None else spec.ray_address
    )
    core_config = spec.core_config(resolved_ray_address=ray_address)
    trainer_overrides = spec.trainer_overrides(ray_address)
    native_placement = trainer_config is not None or any(
        "trainer.placement_group_node_ids." in value for value in trainer_overrides
    )
    if native_placement:
        from verl.single_controller.ray.base import RayResourcePool

        if "placement_group_node_ids" not in signature(RayResourcePool).parameters:
            raise RuntimeError("install the paired verl resource-pool patch before elastic launch")
    if trainer_config is None:
        validate_verl_config(spec, ray_address, trainer_overrides, core_config)
    else:
        validate_verl_config(
            spec, ray_address, trainer_overrides, core_config, config=trainer_config
        )

    def trainer_command(temp):
        trainer_args = trainer_overrides
        if trainer_config is not None:
            from omegaconf import OmegaConf

            OmegaConf.save(trainer_config, Path(temp) / "trainer.yaml", resolve=True)
            trainer_args = ["--config-path", temp, "--config-name", "trainer"]
        return [sys.executable, "-m", "verl.trainer.main_ppo", *trainer_args]

    return run_owned_children(
        spec,
        core_config,
        ray_address,
        trainer_command,
        report_path=report_path,
        wait_first_weights=native_placement,
    )


def run_owned_children(
    spec,
    core_config,
    ray_address,
    trainer_command,
    *,
    report_path=None,
    wait_first_weights=True,
    extra_env=None,
):
    """Own Core, original training command and scaler with common cleanup guarantees."""
    client = ControlClient(
        spec.endpoint,
        spec.deployment_id,
        token=(extra_env or {}).get(
            "ELASTIC_ROLLOUT_TOKEN", os.environ.get("ELASTIC_ROLLOUT_TOKEN")
        ),
        timeout=2,
    )
    env = os.environ.copy()
    env.update(extra_env or {})
    env["RAY_ADDRESS"] = ray_address
    env["ELASTIC_ROLLOUT_EXPECTED_GCS_ADDRESS"] = ray_address
    core = None
    trainer = None
    scaler = None
    report = (
        {
            "ray_gcs_address": ray_address,
            "deployment_id": spec.deployment_id,
            "initial_version": spec.initial_version,
        }
        if report_path is not None
        else None
    )
    interrupted = [0]
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, lambda number, _frame: interrupted.__setitem__(0, number))
    failed = False
    core_cleanup_ok = False
    try:
        with tempfile.TemporaryDirectory(prefix="elastic-rollout-") as temp:
            # An old control server on this endpoint must never satisfy our
            # readiness barrier before the new serve child has bound its port.
            family = socket.AF_INET6 if ":" in spec.bind_host else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                # Match the HTTP server's Linux TIME_WAIT reuse on a quick
                # restart; a live listener still makes bind fail.
                if os.name == "posix":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((spec.bind_host, spec.port))
            config_path = Path(temp) / "core.json"
            config_path.write_text(json.dumps(core_config), encoding="utf-8")
            core = subprocess.Popen(
                owned_command(
                    [
                        sys.executable,
                        "-m",
                        "elastic_rollout.cli",
                        "serve",
                        "--config",
                        str(config_path),
                        "--host",
                        spec.bind_host,
                        "--port",
                        str(spec.port),
                        "--ray-address",
                        ray_address,
                        "--expected-gcs-address",
                        ray_address,
                        "--shutdown-timeout",
                        str(spec.shutdown_timeout_s),
                    ]
                ),
                env=env,
                start_new_session=os.name == "posix",
            )
            ready_status = _wait_ready(client, spec, core_config, core, interrupted)
            logger.info(
                "Core serving_ready: deployment=%s version=%s replicas=%s GCS=%s",
                spec.deployment_id,
                ready_status["serving_policy_version"],
                ready_status["ready_replicas"],
                ray_address,
            )
            if interrupted[0]:
                raise InterruptedError("launch interrupted before trainer start")
            trainer = subprocess.Popen(
                owned_command(trainer_command(temp)),
                env=env,
                start_new_session=os.name == "posix",
            )
            logger.info("started trainer pid=%s on Ray GCS %s", trainer.pid, ray_address)
            # Bound native PG/trainer initialization without dummy reservation actors.
            starting = wait_first_weights
            startup_deadline = time.monotonic() + spec.startup_timeout_s
            while True:
                if interrupted[0]:
                    logger.info(
                        "launcher received signal %s; stopping trainer first", interrupted[0]
                    )
                    failed = True
                    break
                core_code = core.poll()
                trainer_code = trainer.poll()
                if core_code is not None:
                    logger.error("Core exited while trainer was running: %s", core_code)
                    failed = True
                    break
                if trainer_code is not None:
                    if trainer_code:
                        logger.error("trainer exited with status %s", trainer_code)
                        failed = True
                    break
                if starting or (spec.autoscaling is not None and scaler is None):
                    try:
                        updated = client.status().get("serving_policy_version", 0)
                    except (ConnectionError, TimeoutError, OSError):
                        updated = spec.initial_version
                    if updated > spec.initial_version:
                        starting = False
                    elif starting and time.monotonic() > startup_deadline:
                        raise TimeoutError(
                            "Trainer did not commit its first weights before startup timeout"
                        )
                    if (
                        spec.autoscaling is not None
                        and not starting
                        and updated > spec.initial_version
                    ):
                        policy_path = Path(temp) / "autoscaling.json"
                        policy_path.write_text(json.dumps(spec.autoscaling), encoding="utf-8")
                        scaler = subprocess.Popen(
                            owned_command(
                                [
                                    sys.executable,
                                    "-m",
                                    "elastic_rollout.cli",
                                    "autoscale",
                                    "--endpoint",
                                    spec.endpoint,
                                    "--deployment",
                                    spec.deployment_id,
                                    "--config",
                                    str(policy_path),
                                    "--activate",
                                ]
                            ),
                            env=env,
                            start_new_session=os.name == "posix",
                        )
                if scaler is not None and scaler.poll() is not None:
                    raise RuntimeError("elastic autoscaler exited while training was running")
                time.sleep(0.2)
    except (Exception, KeyboardInterrupt):
        logger.exception("owned training launch failed")
        failed = True
    finally:
        cleanup_ok = True
        ports_closed = None
        trainer_code = core_code = None
        try:
            try:
                if scaler is not None and _stop_child(scaler, 30) is None:
                    cleanup_ok = False
            except Exception:
                cleanup_ok = False
                logger.exception("autoscaler termination failed; still stopping Trainer/Core")
            try:
                trainer_code = _stop_child(trainer, spec.trainer_shutdown_timeout_s)
                if trainer is not None and trainer_code is None:
                    cleanup_ok = False
            except Exception:
                cleanup_ok = False
                logger.exception("trainer termination failed; still attempting Core cleanup")
            try:
                core_code = _stop_child(core, spec.shutdown_timeout_s + 30)
                core_cleanup_ok = core is not None and core_code == 0
                if core is not None and not core_cleanup_ok:
                    cleanup_ok = False
                    logger.error("Core graceful cleanup failed or timed out (exit %s)", core_code)
            except Exception:
                cleanup_ok = False
                logger.exception("Core termination failed")
            if core is not None:
                from elastic_rollout.launcher.network import rollout_ports_closed

                try:
                    ports_closed = (
                        None if core_config.get("resource_backend") == "ray_pg"
                        else rollout_ports_closed(core_config["slots"])
                    )
                    if ports_closed is False:
                        cleanup_ok = False
                except Exception:
                    cleanup_ok = False
                    logger.exception("rollout port cleanup check failed")
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        failed = failed or not cleanup_ok
        if report is not None:
            report.update(
                trainer_exit_code=trainer_code,
                core_exit_code=core_code,
                core_graceful_cleanup=core_cleanup_ok,
                rollout_ports_closed=ports_closed,
                interrupted_signal=interrupted[0],
                cleanup_success=cleanup_ok,
            )
    if report_path is not None and report is not None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if interrupted[0]:
        return 128 + interrupted[0] if cleanup_ok else 1
    return 1 if failed else 0
