import os
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
    launcher.start("r", replica.resource, LaunchPlan(("python", "-m", "vllm")))
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
