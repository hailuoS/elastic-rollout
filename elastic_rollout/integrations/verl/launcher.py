"""Process coordination for the static Verl V1 separate_async integration."""

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from elastic_rollout.client import ControlClient
from elastic_rollout.generation import GenerationClient

from .launch_spec import LaunchSpec, validate_hydra_config

logger = logging.getLogger(__name__)


def resolve_ray_cluster(spec: LaunchSpec, core_config: dict) -> str:
    """Attach once, verify capacity, and return the concrete GCS address.

    Ray Client addresses do not expose a direct GCS address for the child
    drivers. The MVP-2B direct actor/HCCL path therefore requires native Ray.
    """
    if spec.ray_address.startswith("ray://") or spec.ray_address == "local":
        raise ValueError("MVP-2B requires an existing native Ray cluster address")
    import ray

    if ray.is_initialized():
        raise RuntimeError("launcher must resolve Ray before another driver initializes it")
    context = ray.init(address=spec.ray_address)
    try:
        address = context.address_info.get("gcs_address")
        if not address or address in ("auto", "local"):
            raise RuntimeError("Ray did not return a concrete GCS address")
        nodes = {node["NodeID"]: node for node in ray.nodes() if node.get("Alive")}
        slots_by_node: dict[str, int] = {}
        for slot in core_config["slots"]:
            node_id = slot["node_id"]
            if node_id not in nodes:
                raise ValueError(f"rollout slot refers to missing Ray node {node_id}")
            slots_by_node[node_id] = slots_by_node.get(node_id, 0) + slot["num_devices"]
        for node_id, count in slots_by_node.items():
            available = nodes[node_id].get("Resources", {}).get("NPU", 0)
            if count > available:
                raise ValueError(
                    f"rollout slots need {count} NPUs on {node_id}, Ray has {available}"
                )
        total = sum(node.get("Resources", {}).get("NPU", 0) for node in nodes.values())
        needed = sum(slots_by_node.values()) + spec.trainer_nodes * spec.trainer_npus_per_node
        if total < needed:
            raise ValueError(
                f"Ray cluster has {total} NPUs; static rollout + trainer need {needed}"
            )
        trainer_nodes_available = sum(
            node.get("Resources", {}).get("NPU", 0) - slots_by_node.get(node_id, 0)
            >= spec.trainer_npus_per_node
            for node_id, node in nodes.items()
        )
        if trainer_nodes_available < spec.trainer_nodes:
            raise ValueError("Ray nodes cannot fit trainer NPUs after rollout slot reservation")
        logger.info("resolved Ray cluster GCS %s with %d live nodes", address, len(nodes))
        return address
    finally:
        ray.shutdown()


def serving_ready(status: dict, spec: LaunchSpec) -> bool:
    """Validate the Core read-only view and the launch identity."""
    deployment = status.get("spec", {})
    replica = deployment.get("replica", {})
    if deployment.get("deployment_id") != spec.deployment_id:
        raise RuntimeError("Core deployment identity differs from LaunchSpec")
    if replica.get("model") != spec.model_path or replica.get("weight_sync") != "hccl":
        raise RuntimeError("Core bootstrap model or transport differs from LaunchSpec")
    if replica.get("policy_version") != spec.expected_version:
        raise RuntimeError("Core bootstrap replica version differs from LaunchSpec")
    bootstrap = status.get("bootstrap_snapshot", {})
    if (
        bootstrap.get("model_path") != spec.model_path
        or bootstrap.get("version") != spec.expected_version
    ):
        raise RuntimeError("Core bootstrap snapshot differs from LaunchSpec")
    if status.get("desired_replicas") != spec.replicas:
        raise RuntimeError("Core desired replica count differs from LaunchSpec")
    version = status.get("serving_policy_version")
    if version != spec.expected_version:
        raise RuntimeError(f"Core serving version {version!r} differs from bootstrap")
    return bool(
        status.get("serving_ready_for_external_update")
        and status.get("ready_replicas") == spec.replicas
        and status.get("accepting_routes")
        and status.get("policy_update") is None
    )


def _stop_child(child: subprocess.Popen | None, timeout: float) -> int | None:
    if child is None:
        return None
    code = child.poll()
    if code is not None:
        return code
    child.terminate()
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("child %s did not exit within %.1fs; forcing exit", child.pid, timeout)
        child.kill()
        child.wait()
        return None


def _wait_ready(client: ControlClient, spec: LaunchSpec, core: subprocess.Popen, stop) -> dict:
    deadline = time.monotonic() + spec.startup_timeout_s
    last_error = "Core control API unavailable"
    while time.monotonic() < deadline:
        if stop[0]:
            raise InterruptedError("launch interrupted before serving readiness")
        if core.poll() is not None:
            raise RuntimeError(f"Core exited before serving readiness (exit {core.returncode})")
        try:
            status = client.status()
            if serving_ready(status, spec):
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


def _smoke_probe(client: ControlClient, spec: LaunchSpec, core: subprocess.Popen, stop) -> dict:
    if not spec.smoke_prompt_ids:
        raise ValueError("--smoke-report requires smoke.prompt_ids in LaunchSpec")
    deadline = time.monotonic() + spec.startup_timeout_s
    while time.monotonic() < deadline:
        if stop[0]:
            raise InterruptedError("launch interrupted before post-update generation")
        if core.poll() is not None:
            raise RuntimeError("Core exited before post-update generation")
        status = client.status()
        version = status.get("serving_policy_version")
        if version is not None and version > spec.expected_version:
            if status.get("serving_ready_for_external_update"):
                break
        time.sleep(0.2)
    else:
        raise TimeoutError(
            "trainer finished without a stable committed policy newer than bootstrap"
        )
    model_name = status["spec"]["replica"]["served_model_name"]
    completion = asyncio.run(
        GenerationClient(client, model_name, request_timeout=60).generate(
            "mvp2b-post-update-smoke",
            list(spec.smoke_prompt_ids),
            {"max_tokens": 4, "temperature": 0},
        )
    )
    if not completion.token_ids or any(item != version for item in completion.versions):
        raise RuntimeError("post-update generation did not use the committed serving version")
    return {
        "initial_version": spec.expected_version,
        "final_version": version,
        "post_update_token_versions": completion.versions,
        "generated_tokens": len(completion.token_ids),
    }


def _rollout_ports_closed(slots: list[dict], timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        open_slots = []
        for slot in slots:
            try:
                with socket.create_connection((slot["host"], slot["port"]), timeout=0.5):
                    open_slots.append(slot["slot_id"])
            except OSError:
                pass
        if not open_slots:
            return True
        if time.monotonic() >= deadline:
            logger.error("rollout serving ports remain open after Core close: %s", open_slots)
            return False
        time.sleep(0.2)


def run_launch(path: Path, *, smoke_report: Path | None = None) -> int:
    spec = LaunchSpec.from_file(path)
    core_config = spec.core_config()
    # Resolve and verify every input before spawning either child.
    ray_address = resolve_ray_cluster(spec, core_config)
    core_config = spec.core_config(resolved_ray_address=ray_address)
    trainer_overrides = spec.trainer_overrides(ray_address)
    validate_hydra_config(trainer_overrides, spec._owned_overrides(ray_address))
    client = ControlClient(
        spec.endpoint, spec.deployment_id, token=os.environ.get("ELASTIC_ROLLOUT_TOKEN"), timeout=2
    )
    env = os.environ.copy()
    env["RAY_ADDRESS"] = ray_address
    env["ELASTIC_ROLLOUT_EXPECTED_GCS_ADDRESS"] = ray_address
    core = None
    trainer = None
    report = None
    interrupted = [0]
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, lambda number, _frame: interrupted.__setitem__(0, number))
    failed = False
    core_cleanup_ok = False
    try:
        with tempfile.TemporaryDirectory(prefix="elastic-rollout-v1-") as temp:
            # An old control server on this endpoint must never satisfy our
            # readiness barrier before the new serve child has bound its port.
            family = socket.AF_INET6 if ":" in spec.bind_host else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.bind((spec.bind_host, spec.port))
            config_path = Path(temp) / "core.json"
            config_path.write_text(json.dumps(core_config), encoding="utf-8")
            core = subprocess.Popen(
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
                ],
                env=env,
            )
            ready_status = _wait_ready(client, spec, core, interrupted)
            logger.info(
                "Core serving_ready: deployment=%s version=%s replicas=%s GCS=%s",
                spec.deployment_id,
                ready_status["serving_policy_version"],
                spec.replicas,
                ray_address,
            )
            if interrupted[0]:
                raise InterruptedError("launch interrupted before trainer start")
            trainer = subprocess.Popen(
                [sys.executable, "-m", "verl.trainer.main_ppo", *trainer_overrides], env=env
            )
            logger.info("started Verl trainer pid=%s on Ray GCS %s", trainer.pid, ray_address)
            while True:
                if interrupted[0]:
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
                time.sleep(0.2)
            if not failed and smoke_report is not None:
                report = _smoke_probe(client, spec, core, interrupted)
                report["ray_gcs_address"] = ray_address
                report["deployment_id"] = spec.deployment_id
                report["initial_ready_replicas"] = ready_status["ready_replicas"]
    except (Exception, KeyboardInterrupt):
        logger.exception("MVP-2B launch failed")
        failed = True
    finally:
        trainer_code = _stop_child(trainer, spec.trainer_shutdown_timeout_s)
        if trainer is not None and trainer_code is None:
            failed = True
        core_code = _stop_child(core, spec.shutdown_timeout_s + 30)
        core_cleanup_ok = core is not None and core_code == 0
        if core is not None and not core_cleanup_ok:
            failed = True
            logger.error("Core graceful cleanup failed or timed out (exit %s)", core_code)
        if report is not None:
            report["rollout_ports_closed"] = _rollout_ports_closed(core_config["slots"])
            if not report["rollout_ports_closed"]:
                failed = True
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if interrupted[0]:
        return 128 + interrupted[0]
    if smoke_report is not None and report is not None:
        report["trainer_exit_code"] = trainer.returncode
        report["core_exit_code"] = core.returncode
        report["core_graceful_cleanup"] = core_cleanup_ok
        smoke_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0
