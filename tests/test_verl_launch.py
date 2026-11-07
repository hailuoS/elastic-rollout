import json
import signal
import socket
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_runtime import ready

from elastic_rollout import cli
from elastic_rollout.api import ReplicaState as S
from elastic_rollout.integrations.verl import launcher
from elastic_rollout.integrations.verl.launch_spec import LaunchSpec, validate_verl_config
from elastic_rollout.launcher import network


def test_megatron_launch_uses_native_config_and_rejects_fsdp_overlap(launch_file):
    _edit_json(launch_file, lambda data: data["verl"].update(strategy="megatron"))
    spec = LaunchSpec.from_file(launch_file)
    with pytest.raises(ValueError, match="conflicts"):
        spec.trainer_overrides("10.0.0.3:6379")
    spec = replace(
        spec,
        user_overrides=tuple(v for v in spec.user_overrides if "actor.strategy=" not in v),
    )
    overrides = spec.trainer_overrides("10.0.0.3:6379")
    assert "model_engine=megatron" in overrides
    assert "actor_rollout_ref.actor.megatron.dtype=bfloat16" in overrides
    assert "actor_rollout_ref.actor.megatron.vanilla_mbridge=false" in overrides
    assert not any("fsdp_config" in value for value in overrides)


@pytest.mark.parametrize(
    "extra,error",
    [
        ([], None),
        (["actor_rollout_ref.actor.megatron.expert_model_parallel_size=2"], "dense BF16"),
        (["actor_rollout_ref.model.lora.rank=8"], "dense BF16"),
    ],
)
def test_megatron_real_hydra_preflight(launch_file, monkeypatch, extra, error):
    pytest.importorskip("hydra")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "verl"))
    spec = replace(LaunchSpec.from_file(launch_file), strategy="megatron")
    (Path(spec.model_path) / "config.json").write_text('{"model_type":"qwen2"}')
    spec = replace(
        spec,
        user_overrides=tuple(v for v in spec.user_overrides if "actor.strategy=" not in v)
        + tuple(extra),
    )
    address = "10.0.0.3:6379"
    args = (spec, address, spec.trainer_overrides(address), spec.core_config())
    if error:
        with pytest.raises(ValueError, match=error):
            validate_verl_config(*args)
    else:
        validate_verl_config(*args)


@pytest.fixture
def launch_file(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    template = tmp_path / "core.json"
    template.write_text(
        json.dumps(
            {
                "slots": [
                    {
                        "slot_id": "rollout-0",
                        "node_id": "node-0",
                        "host": "10.0.0.2",
                        "num_devices": 2,
                        "port": 9001,
                    }
                ],
                "deployments": [{"desired_replicas": 1, "replica": {"tensor_parallel_size": 2}}],
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "launch.json"
    path.write_text(
        json.dumps(
            {
                "runtime": {"ray_address": "auto"},
                "controller": {
                    "bind_host": "127.0.0.1",
                    "advertise_host": "10.0.0.1",
                    "port": 18080,
                },
                "deployment": {"id": "rollout", "core_template": "core.json"},
                "model": {"path": str(model), "initial_version": 0},
                "verl": {
                    "overrides": [
                        "data.train_files=/tmp/train.parquet",
                        "data.train_batch_size=1",
                        "trainer.device=npu",
                        "trainer.resume_mode=disable",
                        "trainer.nnodes=1",
                        "trainer.n_gpus_per_node=2",
                        "trainer.v1.separate_async.parameter_sync_step=1",
                        "actor_rollout_ref.actor.strategy=fsdp2",
                        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
                        "actor_rollout_ref.rollout.name=vllm",
                        "actor_rollout_ref.rollout.mode=async",
                        "actor_rollout_ref.rollout.nnodes=1",
                        "actor_rollout_ref.rollout.n_gpus_per_node=2",
                        "actor_rollout_ref.rollout.free_cache_engine=false",
                        "actor_rollout_ref.rollout.multi_turn.enable=false",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _edit_json(path, edit):
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_launch_spec_owns_core_and_verl_identity(launch_file):
    spec = LaunchSpec.from_file(launch_file)
    core = spec.core_config()
    assert core["deployments"][0]["replica"]["model"] == spec.model_path
    assert core["deployments"][0]["replica"]["weight_sync"] == "hccl"
    overrides = spec.trainer_overrides("10.0.0.3:6379")
    assert "+ray_kwargs.ray_init.address=10.0.0.3:6379" in overrides
    assert f"actor_rollout_ref.model.path={spec.model_path}" in overrides
    assert f"++actor_rollout_ref.rollout.custom.elastic.endpoint={spec.endpoint}" in overrides
    assert "actor_rollout_ref.rollout.nnodes=1" in overrides
    assert "trainer.n_gpus_per_node=2" in overrides
    assert "trainer.n_gpus_per_node" not in spec.integration_values("10.0.0.3:6379")
    assert "data.train_files=/tmp/train.parquet" in overrides
    with pytest.raises(ValueError, match="concrete"):
        spec.trainer_overrides("auto")


def test_generated_overrides_compose_real_verl_base(launch_file, monkeypatch):
    pytest.importorskip("hydra")
    verl_checkout = Path(__file__).resolve().parents[2] / "verl"
    if not verl_checkout.is_dir():
        pytest.skip("adjacent verl checkout unavailable")
    monkeypatch.syspath_prepend(str(verl_checkout))
    spec = LaunchSpec.from_file(launch_file)
    address = "10.0.0.3:6379"
    validate_verl_config(spec, address, spec.trainer_overrides(address), spec.core_config())


@pytest.mark.parametrize("policy_mode", ["service", "pipeline"])
def test_elastic_launch_only_installs_observer_for_pipeline_mode(
    launch_file, monkeypatch, policy_mode
):
    pytest.importorskip("hydra")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "verl"))
    template = launch_file.parent / "core.json"

    def edit_core(data):
        data["deployments"][0]["elastic_scaling"] = True
        data["slots"].append(dict(data["slots"][0], slot_id="rollout-1", port=9002))

    _edit_json(template, edit_core)

    def edit_launch(data):
        data["verl"]["autoscaling"] = {"policy_mode": policy_mode}
        data["verl"]["overrides"] = [
            "actor_rollout_ref.rollout.n_gpus_per_node=4"
            if item.startswith("actor_rollout_ref.rollout.n_gpus_per_node=")
            else item
            for item in data["verl"]["overrides"]
        ]

    _edit_json(launch_file, edit_launch)
    spec = LaunchSpec.from_file(launch_file)
    address = "10.0.0.3:6379"
    overrides = spec.trainer_overrides(address)
    assert any("ElasticPipelineReplayBuffer" in item for item in overrides) == (
        policy_mode == "pipeline"
    )
    validate_verl_config(spec, address, overrides, spec.core_config())


def test_model_path_with_spaces_is_quoted_for_real_hydra(launch_file, monkeypatch):
    pytest.importorskip("hydra")
    verl_checkout = Path(__file__).resolve().parents[2] / "verl"
    if not verl_checkout.is_dir():
        pytest.skip("adjacent verl checkout unavailable")
    monkeypatch.syspath_prepend(str(verl_checkout))
    model = Path(json.loads(launch_file.read_text(encoding="utf-8"))["model"]["path"])
    renamed = model.rename(model.parent / "model with spaces")
    _edit_json(launch_file, lambda data: data["model"].update(path=str(renamed)))
    spec = LaunchSpec.from_file(launch_file)
    address = "10.0.0.3:6379"
    validate_verl_config(spec, address, spec.trainer_overrides(address), spec.core_config())


@pytest.mark.parametrize(
    "field,value",
    [
        ("deployment_id", "other"),
        ("replica.model", "/wrong/model"),
        ("replica.policy_version", 1),
        ("replica.weight_sync", "snapshot"),
    ],
)
def test_core_template_conflicts_fail(launch_file, field, value):
    template = launch_file.parent / "core.json"

    def edit(data):
        target = data["deployments"][0]
        parts = field.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    _edit_json(template, edit)
    with pytest.raises(ValueError, match="conflicts"):
        LaunchSpec.from_file(launch_file).core_config()


@pytest.mark.parametrize(
    "override",
    [
        "actor_rollout_ref.model.path=/other",
        "++actor_rollout_ref.rollout.custom.elastic.endpoint=http://other:8000",
        "ray_kwargs.ray_init.address=auto",
        "++actor_rollout_ref.rollout.custom.elastic={endpoint:wrong}",
    ],
)
def test_verl_override_conflicts_fail(launch_file, override):
    _edit_json(launch_file, lambda data: data["verl"].update(overrides=[override]))
    with pytest.raises(ValueError, match="conflicts|overlaps"):
        LaunchSpec.from_file(launch_file).trainer_overrides("10.0.0.3:6379")


def test_trainer_resource_settings_remain_verl_owned(launch_file, monkeypatch):
    pytest.importorskip("hydra")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "verl"))
    _edit_json(
        launch_file,
        lambda data: data["verl"]["overrides"].append("trainer.n_gpus_per_node=8"),
    )
    spec = LaunchSpec.from_file(launch_file)
    address = "10.0.0.3:6379"
    validate_verl_config(spec, address, spec.trainer_overrides(address), spec.core_config())


@pytest.mark.parametrize(
    "override,error",
    [
        ("data.train_batch_size=2", "train_batch_size"),
        ("actor_rollout_ref.rollout.n_gpus_per_node=4", "resource shape"),
    ],
)
def test_verl_owned_training_values_still_validate(launch_file, monkeypatch, override, error):
    pytest.importorskip("hydra")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "verl"))
    _edit_json(launch_file, lambda data: data["verl"]["overrides"].append(override))
    spec = LaunchSpec.from_file(launch_file)
    address = "10.0.0.3:6379"
    with pytest.raises(ValueError, match=error):
        validate_verl_config(spec, address, spec.trainer_overrides(address), spec.core_config())


def test_static_slot_and_unknown_endpoint_conflicts_fail(launch_file):
    template = launch_file.parent / "core.json"
    _edit_json(template, lambda data: data.update(endpoint="http://other:8080"))
    with pytest.raises(ValueError, match="endpoint conflicts"):
        LaunchSpec.from_file(launch_file).core_config()
    _edit_json(template, lambda data: data.pop("endpoint"))
    _edit_json(template, lambda data: data.update(slots=[]))
    with pytest.raises(ValueError, match="static rollout slots"):
        LaunchSpec.from_file(launch_file).core_config()


def test_core_template_ray_address_checks_resolved_identity(launch_file):
    template = launch_file.parent / "core.json"
    _edit_json(template, lambda data: data.update(ray_address="10.0.0.3:6379"))
    spec = LaunchSpec.from_file(launch_file)
    assert spec.core_config(resolved_ray_address="10.0.0.3:6379")
    with pytest.raises(ValueError, match="resolved Ray cluster"):
        spec.core_config(resolved_ray_address="10.0.0.4:6379")


def test_unknown_launch_fields_fail_fast(launch_file):
    _edit_json(launch_file, lambda data: data["controller"].update(endpoint="http://wrong"))
    with pytest.raises(ValueError, match="unsupported controller fields"):
        LaunchSpec.from_file(launch_file)


def test_fixed_verl_base_config_is_not_public(launch_file):
    _edit_json(launch_file, lambda data: data["verl"].update(base_config="ppo_trainer"))
    with pytest.raises(ValueError, match="unsupported verl fields"):
        LaunchSpec.from_file(launch_file)


def test_cleanup_detects_open_rollout_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        slot = {"host": "127.0.0.1", "port": listener.getsockname()[1], "slot_id": "r0"}
        assert not network.rollout_ports_closed([slot], timeout=0.01)
    assert network.rollout_ports_closed([slot], timeout=0.01)


def test_ready_view_reuses_policy_topology_gate(runtime):
    _, manager, _, _ = runtime
    ready(runtime)
    manager.spec = replace(manager.spec, replica=replace(manager.spec.replica, weight_sync="hccl"))
    assert manager.status()["serving_ready_for_external_update"] is True
    manager.traffic.replicas["terminating"] = SimpleNamespace(state=S.FAILED, resource=object())
    assert manager.status()["ready_replicas"] == manager.desired
    assert manager.status()["serving_ready_for_external_update"] is False
    del manager.traffic.replicas["terminating"]
    manager.traffic.pause_routes("test")
    assert manager.status()["serving_ready_for_external_update"] is False


def test_launcher_validates_bootstrap_once_and_uses_core_readiness(launch_file):
    spec = LaunchSpec.from_file(launch_file)
    status = {
        "spec": {
            "deployment_id": spec.deployment_id,
            "replica": {
                "model": spec.model_path,
                "policy_version": 0,
                "weight_sync": "hccl",
            },
        },
        "bootstrap_snapshot": {"model_path": spec.model_path, "version": 0},
        "desired_replicas": 1,
        "ready_replicas": 1,
        "serving_policy_version": 0,
        "accepting_routes": True,
        "serving_ready_for_external_update": True,
        "policy_update": None,
    }
    launcher.validate_bootstrap(status, spec, spec.core_config())
    status["serving_ready_for_external_update"] = False
    assert not status["serving_ready_for_external_update"]
    status["spec"]["replica"]["model"] = "/wrong/model"
    with pytest.raises(RuntimeError, match="bootstrap model"):
        launcher.validate_bootstrap(status, spec, spec.core_config())


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("serving_policy_version", 1, "serving version changed"),
        ("desired_replicas", 2, "desired topology changed"),
    ],
)
def test_readiness_rejects_identity_change_before_trainer(
    launch_file, monkeypatch, field, value, error
):
    spec = LaunchSpec.from_file(launch_file)
    status = {
        "spec": {
            "deployment_id": spec.deployment_id,
            "replica": {"model": spec.model_path, "policy_version": 0, "weight_sync": "hccl"},
        },
        "bootstrap_snapshot": {"model_path": spec.model_path, "version": 0},
        "desired_replicas": 1,
        "serving_policy_version": 0,
        "serving_ready_for_external_update": False,
    }
    calls = 0

    def next_status():
        nonlocal calls
        calls += 1
        if calls == 2:
            status["serving_ready_for_external_update"] = True
            status[field] = value
        return status

    monkeypatch.setattr(launcher.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match=error):
        launcher._wait_ready(
            SimpleNamespace(status=next_status),
            spec,
            spec.core_config(),
            SimpleNamespace(poll=lambda: None),
            [0],
        )


def test_ray_auto_checks_rollout_slots_without_predicting_trainer_placement(
    launch_file, monkeypatch
):
    spec = LaunchSpec.from_file(launch_file)
    shutdown = Mock()
    ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: SimpleNamespace(address_info={"gcs_address": "10.0.0.3:6379"}),
        nodes=lambda: [{"NodeID": "node-0", "Alive": True, "Resources": {"NPU": 2}}],
        shutdown=shutdown,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    assert launcher.resolve_ray_cluster(spec, spec.core_config()) == "10.0.0.3:6379"
    shutdown.assert_called_once()
    core = spec.core_config()
    core["slots"][0]["num_devices"] = 3
    with pytest.raises(ValueError, match="rollout slots need 3"):
        launcher.resolve_ray_cluster(spec, core)
    assert shutdown.call_count == 2


def test_serve_reinstalls_sigterm_after_ray_initialization(launch_file, monkeypatch):
    template = LaunchSpec.from_file(launch_file).core_config()
    template_path = launch_file.parent / "full-core.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    original = signal.getsignal(signal.SIGTERM)
    events = []
    controller = Mock()
    controller.ensure_deployment.side_effect = lambda *_: events.append("ensure")
    controller.close.side_effect = lambda **_: events.append("close")
    server = Mock()

    def serve_forever(**_):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler != signal.SIG_DFL
        events.append("signal")
        handler(signal.SIGTERM, None)

    server.serve_forever.side_effect = serve_forever
    monkeypatch.setattr(cli, "Controller", lambda *_: controller)
    monkeypatch.setattr(cli, "make_server", lambda *_: server)
    monkeypatch.setattr(cli, "Reconciler", lambda *_: Mock())

    def fake_ray_launcher(*_args, **_kwargs):
        events.append("ray.init")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        return Mock()

    import elastic_rollout.launcher.ray as ray_launcher

    monkeypatch.setattr(ray_launcher, "RayProcessLauncher", fake_ray_launcher)
    monkeypatch.setattr(sys, "argv", ["elastic-rollout", "serve", "--config", str(template_path)])
    try:
        cli.main()
    finally:
        signal.signal(signal.SIGTERM, original)
    assert events == ["ray.init", "ensure", "signal", "close"]
    server.shutdown.assert_called_once()


@pytest.mark.parametrize(
    "core_exit,trainer_exit,cleanup_exit,expected",
    [
        (None, 0, 0, 0),
        (1, None, 1, 1),
        (None, 2, 0, 1),
        (None, 0, 1, 1),
    ],
)
def test_launcher_supervises_both_children(
    launch_file, monkeypatch, core_exit, trainer_exit, cleanup_exit, expected
):
    class Child:
        def __init__(self, first, cleanup):
            self.first, self.cleanup = first, cleanup
            self.pid = 42
            self.returncode = None
            self.terminated = False

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            return self.first

        def terminate(self):
            self.terminated = True
            self.returncode = self.cleanup

        def wait(self, timeout=None):
            return self.returncode

    core = Child(core_exit, cleanup_exit)
    trainer = Child(trainer_exit, 0)
    processes = iter((core, trainer))
    monkeypatch.setattr(launcher, "resolve_ray_cluster", lambda *_: "10.0.0.3:6379")
    monkeypatch.setattr(launcher, "validate_verl_config", lambda *_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: next(processes))
    monkeypatch.setattr(
        launcher, "_wait_ready", lambda *a: {"serving_policy_version": 0, "ready_replicas": 1}
    )
    monkeypatch.setattr(
        launcher.socket,
        "socket",
        lambda *_: nullcontext(SimpleNamespace(bind=lambda *_: None, setsockopt=lambda *_: None)),
    )
    monkeypatch.setattr(launcher.time, "sleep", lambda *_: None)
    monkeypatch.setattr(network, "rollout_ports_closed", lambda *_: True)
    assert launcher.run_launch(launch_file) == expected
    if core_exit is None:
        assert core.terminated
    if trainer_exit is None:
        assert trainer.terminated


def test_bootstrap_failure_never_starts_trainer(launch_file, monkeypatch):
    child = Mock()
    child.poll.return_value = None
    child.wait.return_value = 0
    child.pid = 42
    starts = Mock(return_value=child)
    monkeypatch.setattr(launcher, "resolve_ray_cluster", lambda *_: "10.0.0.3:6379")
    monkeypatch.setattr(launcher, "validate_verl_config", lambda *_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", starts)
    monkeypatch.setattr(launcher, "_wait_ready", Mock(side_effect=TimeoutError("not ready")))
    monkeypatch.setattr(network, "rollout_ports_closed", lambda *_: True)
    monkeypatch.setattr(
        launcher.socket,
        "socket",
        lambda *_: nullcontext(SimpleNamespace(bind=lambda *_: None, setsockopt=lambda *_: None)),
    )
    assert launcher.run_launch(launch_file) == 1
    starts.assert_called_once()
    child.terminate.assert_called_once()


@pytest.mark.parametrize("ports_closed,stop_error", [(True, False), (False, False), (True, True)])
def test_interrupted_launch_reports_cleanup_and_always_stops_core(
    launch_file, tmp_path, monkeypatch, ports_closed, stop_error
):
    core, trainer = Mock(pid=101), Mock(pid=102)
    core.poll.return_value = trainer.poll.return_value = None
    children = iter((core, trainer))
    events = []

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"] == (launcher.os.name == "posix")
        child = next(children)
        if child is trainer:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    def stop(child, timeout):
        events.append(child.pid)
        if child is trainer and stop_error:
            raise OSError("termination failed")
        return -15 if child is trainer else 0

    monkeypatch.setattr(launcher, "resolve_ray_cluster", lambda *_: "10.0.0.3:6379")
    monkeypatch.setattr(launcher, "validate_verl_config", lambda *_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher, "_stop_child", stop)
    monkeypatch.setattr(
        launcher, "_wait_ready", lambda *_: {"serving_policy_version": 0, "ready_replicas": 1}
    )
    monkeypatch.setattr(
        launcher.socket,
        "socket",
        lambda *_: nullcontext(SimpleNamespace(bind=lambda *_: None, setsockopt=lambda *_: None)),
    )
    port_check = Mock(return_value=ports_closed)
    monkeypatch.setattr(network, "rollout_ports_closed", port_check)
    original = signal.getsignal(signal.SIGTERM)
    report_path = tmp_path / "run-b.json"
    code = launcher.run_launch(launch_file, report_path=report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert events == [102, 101]
    assert code == (143 if ports_closed and not stop_error else 1)
    assert report["interrupted_signal"] == signal.SIGTERM
    assert report["core_graceful_cleanup"] is True
    assert report["rollout_ports_closed"] is ports_closed
    assert report["cleanup_success"] == (ports_closed and not stop_error)
    assert "generated_tokens" not in report
    assert signal.getsignal(signal.SIGTERM) == original
    port_check.assert_called_once()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux TCP TIME_WAIT semantics")
@pytest.mark.parametrize("live", [False, True])
def test_launch_port_reuses_closed_server_but_rejects_live_listener(launch_file, monkeypatch, live):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        if not live:
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                accepted, _ = server.accept()
                with accepted:
                    accepted.settimeout(2)
                    accepted.shutdown(socket.SHUT_WR)
                    assert client.recv(1) == b""
                    client.shutdown(socket.SHUT_WR)
                    assert accepted.recv(1) == b""
            server.close()
            # Confirm this exercises TIME_WAIT, rather than a plain unused port.
            with socket.socket() as old_probe, pytest.raises(OSError):
                old_probe.bind(("127.0.0.1", port))
        _edit_json(launch_file, lambda data: data["controller"].update(port=port))
        core, trainer = Mock(pid=101), Mock(pid=102)
        core.poll.return_value, trainer.poll.return_value = None, 0
        starts = Mock(side_effect=[core, trainer])
        monkeypatch.setattr(launcher, "resolve_ray_cluster", lambda *_: "10.0.0.3:6379")
        monkeypatch.setattr(launcher, "validate_verl_config", lambda *_: None)
        monkeypatch.setattr(launcher.subprocess, "Popen", starts)
        monkeypatch.setattr(launcher, "_stop_child", lambda *_: 0)
        monkeypatch.setattr(
            launcher, "_wait_ready", lambda *_: {"serving_policy_version": 0, "ready_replicas": 1}
        )
        monkeypatch.setattr(network, "rollout_ports_closed", lambda *_: True)
        assert launcher.run_launch(launch_file) == (1 if live else 0)
        assert starts.call_count == (0 if live else 2)
