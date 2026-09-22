import errno
import os
import socket
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_vllm import replica_and_spec

from elastic_rollout.launcher.base import LaunchPlan
from elastic_rollout.launcher.ray import RayProcessLauncher, _ProcessActor


@pytest.fixture
def fake_ray(monkeypatch):
    ray = ModuleType("ray")
    ray.is_initialized = lambda: True
    ray.remote = MagicMock()
    ray.wait = MagicMock(return_value=([], []))
    ray.get = MagicMock(return_value=())
    ray.get_runtime_context = MagicMock()
    ray.kill = MagicMock()
    ray.exceptions = SimpleNamespace(RayActorError=ConnectionError)
    monkeypatch.setitem(sys.modules, "ray", ray)
    strategies = ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda node_id, **kwargs: dict(
        node_id=node_id, **kwargs
    )
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    return ray


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_ray_reserves_npu_and_node_before_start(fake_ray, tp):
    launcher = RayProcessLauncher()
    replica, _ = replica_and_spec(tp=tp)
    plan = LaunchPlan(("python", "-m", "vllm"), (("VLLM_SERVER_DEV_MODE", "1"),))
    launcher.start("r", replica.resource, plan)
    fake_ray.remote.return_value.options.return_value.remote.return_value.start.remote.assert_called_once_with(
        plan.argv, tp, replica.resource.slot.port, dict(plan.env)
    )
    options = fake_ray.remote.return_value.options.call_args.kwargs
    assert options["resources"] == {"NPU": tp}
    assert options["scheduling_strategy"] == {"node_id": "node", "soft": False}
    assert options["max_restarts"] == 0
    assert not launcher.started("r")
    assert launcher.alive("r")
    launcher.stop("r")
    fake_ray.kill.assert_called_once()
    assert not launcher.actors


def test_unreachable_actor_quarantines_slot(fake_ray):
    launcher = RayProcessLauncher()
    replica, _ = replica_and_spec()
    launcher.start("r", replica.resource, LaunchPlan(("python",)))
    fake_ray.wait.return_value = (["started"], [])
    fake_ray.get.side_effect = ConnectionError("partition")
    with pytest.raises(RuntimeError, match="quarantined"):
        launcher.stop("r")
    assert "r" in launcher.actors
    fake_ray.kill.assert_not_called()


@pytest.mark.parametrize(
    "assigned,expected", [(["5", "4"], "4,5"), (["3", "2"], "2,3"), (["10", "2"], "2,10")]
)
@pytest.mark.parametrize("visible", [None, "0,1,2,3,4,5,6,7", "5,4"])
def test_child_devices_come_from_ray_not_actor_environment(
    fake_ray, monkeypatch, assigned, expected, visible
):
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"NPU": assigned}
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "1")
    monkeypatch.setattr("elastic_rollout.launcher.ray.socket.socket", MagicMock())
    popen = MagicMock()
    monkeypatch.setattr("elastic_rollout.launcher.ray.subprocess.Popen", popen)
    before = dict(os.environ)
    actor = _ProcessActor()
    result = actor.start(("python", "-m", "vllm"), 2, 8000)
    assert result == tuple(expected.split(","))
    assert set(result) == set(assigned)
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.assert_called_once_with()
    child_env = popen.call_args.kwargs["env"]
    assert child_env == before | {"ASCEND_RT_VISIBLE_DEVICES": expected}
    assert dict(os.environ) == before
    assert "elastic_rollout.launcher.supervisor" in popen.call_args.args[0]


@pytest.mark.parametrize(
    "assigned",
    [[], ["0"], ["0", "1", "2"], ["1", "1"], ["0", "00"], ["-1", "2"], ["x", "2"], [0, 1]],
)
def test_bad_assignment_fails_before_launch(fake_ray, monkeypatch, assigned):
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"NPU": assigned}
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    popen = MagicMock()
    monkeypatch.setattr("elastic_rollout.launcher.ray.subprocess.Popen", popen)
    with pytest.raises(RuntimeError, match="expected 2 distinct"):
        _ProcessActor().start(("python",), 2, 8000)
    popen.assert_not_called()


def test_actor_retains_assignment_after_spawn_failure(fake_ray, monkeypatch):
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"NPU": ["5", "4"]}
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    monkeypatch.setattr("elastic_rollout.launcher.ray.socket.socket", MagicMock())
    monkeypatch.setattr(
        "elastic_rollout.launcher.ray.subprocess.Popen",
        MagicMock(side_effect=OSError("spawn failed")),
    )
    actor = _ProcessActor()
    with pytest.raises(OSError, match="spawn failed"):
        actor.start(("python",), 2, 8000)
    assert actor.stop() == ("4", "5")


def test_launch_environment_cannot_override_ray_devices(fake_ray, monkeypatch):
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"NPU": ["5", "4"]}
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    monkeypatch.setattr("elastic_rollout.launcher.ray.socket.socket", MagicMock())
    popen = MagicMock()
    monkeypatch.setattr("elastic_rollout.launcher.ray.subprocess.Popen", popen)
    _ProcessActor().start(
        ("python",),
        2,
        8000,
        {
            "VLLM_SERVER_DEV_MODE": "1",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        },
    )
    env = popen.call_args.kwargs["env"]
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert env["VLLM_SERVER_DEV_MODE"] == "1"


@pytest.mark.parametrize("start_failed", [False, True])
def test_launcher_records_assignment_even_when_start_failed(fake_ray, start_failed):
    launcher = RayProcessLauncher()
    replica, _ = replica_and_spec(tp=2)
    launcher.start("r", replica.resource, LaunchPlan(("python",)))
    fake_ray.wait.return_value = (["started"], [])
    if start_failed:
        fake_ray.get.side_effect = RuntimeError("spawn failed")
        with pytest.raises(RuntimeError, match="spawn failed"):
            launcher.started("r")
        fake_ray.get.side_effect = None
    else:
        fake_ray.get.return_value = ("4", "5")
        assert launcher.started("r")
        assert replica.resource.allocation.actual_device_ids == ("4", "5")
    fake_ray.get.return_value = ("4", "5")
    launcher.stop("r")
    assert replica.resource.allocation.actual_device_ids == ("4", "5")
    assert not launcher.leases


@pytest.mark.parametrize("conflict_at", [None, "bind", "listen"])
def test_port_probe_allows_reuse_but_rejects_conflicts_before_spawn(
    fake_ray, monkeypatch, conflict_at
):
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"NPU": ["0", "1"]}
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    probe = MagicMock()
    factory = MagicMock()
    factory.return_value.__enter__.return_value = probe
    monkeypatch.setattr("elastic_rollout.launcher.ray.socket.socket", factory)
    popen = MagicMock()
    monkeypatch.setattr("elastic_rollout.launcher.ray.subprocess.Popen", popen)

    def bind(address):
        probe.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        assert address == ("0.0.0.0", 8000)
        if conflict_at == "bind":
            raise OSError(errno.EADDRINUSE, "Address already in use")

    probe.bind.side_effect = bind
    if conflict_at == "listen":
        probe.listen.side_effect = OSError(errno.EADDRINUSE, "Address already in use")
    actor = _ProcessActor()
    if conflict_at:
        with pytest.raises(OSError, match="vLLM API port 8000 unavailable") as exc:
            actor.start(("python",), 2, 8000)
        assert exc.value.errno == errno.EADDRINUSE
        popen.assert_not_called()
        assert actor.process is None
    else:
        actor.start(("python",), 2, 8000)
        probe.listen.assert_called_once_with(1)
        popen.assert_called_once()
    factory.return_value.__exit__.assert_called_once()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux TCP port reuse semantics")
def test_real_linux_port_probe_rejects_live_listener(fake_ray, monkeypatch):
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"NPU": ["0", "1"]}
    popen = MagicMock()
    monkeypatch.setattr("elastic_rollout.launcher.ray.subprocess.Popen", popen)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with pytest.raises(OSError) as exc:
            _ProcessActor().start(("python",), 2, port)
        assert exc.value.errno == errno.EADDRINUSE
    popen.assert_not_called()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux TCP TIME_WAIT semantics")
def test_real_linux_port_probe_reuses_closed_server_connections(fake_ray, monkeypatch):
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"NPU": ["0", "1"]}
    popen = MagicMock()
    monkeypatch.setattr("elastic_rollout.launcher.ray.subprocess.Popen", popen)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(2)
        server.bind(("0.0.0.0", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = server.accept()
            connection.close()  # Server actively closes, leaving its port in TIME_WAIT.
            assert client.recv(1) == b""
    # Reproduce the old probe's failure after the listener has already closed.
    with socket.socket() as old_probe, pytest.raises(OSError) as exc:
        old_probe.bind(("0.0.0.0", port))
    assert exc.value.errno == errno.EADDRINUSE
    actor = _ProcessActor()
    assert actor.start(("python",), 2, port) == ("0", "1")
    popen.assert_called_once()
