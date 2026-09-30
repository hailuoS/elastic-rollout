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
from elastic_rollout.integrations.verl.launch_spec import LaunchSpec, validate_hydra_config


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
                "deployments": [{"replica": {"tensor_parallel_size": 2}}],
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
                "deployment": {"id": "rollout", "replicas": 1, "core_template": "core.json"},
                "model": {"path": str(model), "shared_immutable": True, "initial_version": 0},
                "verl": {
                    "base_config": "ppo_trainer",
                    "trainer_nodes": 1,
                    "trainer_npus_per_node": 2,
                    "actor_mini_batch_size": 1,
                    "parameter_sync_step": 1,
                    "overrides": ["data.train_files=/tmp/train.parquet"],
                },
                "smoke": {"prompt_ids": [1, 2]},
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
    assert "actor_rollout_ref.rollout.n_gpus_per_node=2" in overrides
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
    validate_hydra_config(spec.trainer_overrides(address), spec._owned_overrides(address))


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
    validate_hydra_config(spec.trainer_overrides(address), spec._owned_overrides(address))


@pytest.mark.parametrize(
    "field,value",
    [
        ("deployment_id", "other"),
        ("desired_replicas", 2),
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
        "trainer.n_gpus_per_node=8",
        "++actor_rollout_ref.rollout.custom.elastic.endpoint=http://other:8000",
        "ray_kwargs.ray_init.address=auto",
        "++actor_rollout_ref.rollout.custom.elastic={endpoint:wrong}",
    ],
)
def test_verl_override_conflicts_fail(launch_file, override):
    _edit_json(launch_file, lambda data: data["verl"].update(overrides=[override]))
    with pytest.raises(ValueError, match="conflicts|overlaps"):
        LaunchSpec.from_file(launch_file).trainer_overrides("10.0.0.3:6379")


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


def test_smoke_requires_generation_from_committed_version(launch_file, monkeypatch):
    spec = LaunchSpec.from_file(launch_file)
    status = {
        "serving_policy_version": 1,
        "serving_ready_for_external_update": True,
        "spec": {"replica": {"served_model_name": "policy"}},
    }
    client = SimpleNamespace(status=lambda: status)
    core = SimpleNamespace(poll=lambda: None)

    class FakeGenerationClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def generate(self, *_args, **_kwargs):
            return SimpleNamespace(token_ids=[42], versions=[status["serving_policy_version"]])

    monkeypatch.setattr(launcher, "GenerationClient", FakeGenerationClient)
    report = launcher._smoke_probe(client, spec, core, [0])
    assert report["post_update_token_versions"] == [1]
    status["serving_policy_version"] = 0
    with pytest.raises(TimeoutError, match="newer than bootstrap"):
        launcher._smoke_probe(client, replace(spec, startup_timeout_s=0.01), core, [0])


def test_smoke_detects_open_rollout_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        slot = {"host": "127.0.0.1", "port": listener.getsockname()[1], "slot_id": "r0"}
        assert not launcher._rollout_ports_closed([slot], timeout=0.01)
    assert launcher._rollout_ports_closed([slot], timeout=0.01)


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


def test_serving_ready_validates_bootstrap_identity(launch_file):
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
    assert launcher.serving_ready(status, spec)
    status["serving_ready_for_external_update"] = False
    assert not launcher.serving_ready(status, spec)
    status["spec"]["replica"]["model"] = "/wrong/model"
    with pytest.raises(RuntimeError, match="bootstrap model"):
        launcher.serving_ready(status, spec)


def test_ray_auto_is_resolved_and_capacity_checked_before_children(launch_file, monkeypatch):
    spec = LaunchSpec.from_file(launch_file)
    shutdown = Mock()
    ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: SimpleNamespace(address_info={"gcs_address": "10.0.0.3:6379"}),
        nodes=lambda: [{"NodeID": "node-0", "Alive": True, "Resources": {"NPU": 4}}],
        shutdown=shutdown,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    assert launcher.resolve_ray_cluster(spec, spec.core_config()) == "10.0.0.3:6379"
    shutdown.assert_called_once()
    insufficient = replace(spec, trainer_npus_per_node=3)
    with pytest.raises(ValueError, match="need 5"):
        launcher.resolve_ray_cluster(insufficient, insufficient.core_config())
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
    monkeypatch.setattr(launcher, "validate_hydra_config", lambda *_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: next(processes))
    monkeypatch.setattr(launcher, "_wait_ready", lambda *a: {"serving_policy_version": 0})
    monkeypatch.setattr(
        launcher.socket, "socket", lambda *_: nullcontext(SimpleNamespace(bind=lambda *_: None))
    )
    monkeypatch.setattr(launcher.time, "sleep", lambda *_: None)
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
    monkeypatch.setattr(launcher, "validate_hydra_config", lambda *_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", starts)
    monkeypatch.setattr(launcher, "_wait_ready", Mock(side_effect=TimeoutError("not ready")))
    monkeypatch.setattr(
        launcher.socket, "socket", lambda *_: nullcontext(SimpleNamespace(bind=lambda *_: None))
    )
    assert launcher.run_launch(launch_file) == 1
    starts.assert_called_once()
    child.terminate.assert_called_once()
