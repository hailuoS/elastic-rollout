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
    ray.get = MagicMock(return_value=True)
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


@pytest.mark.parametrize("expected", [tuple(map(str, range(8))), ("2", "3")])
def test_process_actor_does_not_override_ray_devices(monkeypatch, expected):
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    popen = MagicMock()
    monkeypatch.setattr("elastic_rollout.launcher.ray.subprocess.Popen", popen)
    with pytest.raises(RuntimeError, match="expected"):
        _ProcessActor().start(("python",), expected, 8000)
    popen.assert_not_called()
