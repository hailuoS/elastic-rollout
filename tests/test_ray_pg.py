import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from test_ray_launcher import fake_ray as ray_fixture
from test_verl_native_entry import native_config
from test_vllm import replica_and_spec

from elastic_rollout.integrations.verl.launch_spec import validate_verl_config
from elastic_rollout.integrations.verl.native_entry import NativeOptions, build_launch
from elastic_rollout.launcher.base import LaunchPlan
from elastic_rollout.launcher.ray_pg import RayPGProcessLauncher

fake_ray = ray_fixture


@pytest.fixture
def pg_api(monkeypatch, fake_ray):
    api = ModuleType("ray.util.placement_group")
    api.placement_group = MagicMock()
    api.remove_placement_group = MagicMock()
    api.placement_group_table = MagicMock(return_value={"state": "REMOVED"})
    monkeypatch.setitem(sys.modules, api.__name__, api)
    sys.modules["ray.util.scheduling_strategies"].PlacementGroupSchedulingStrategy = (
        lambda **kwargs: kwargs
    )
    return api


def test_pending_request_is_idempotent_and_cancellable(fake_ray, pg_api):
    launcher = RayPGProcessLauncher(resource_label="rollout")
    replica, _ = replica_and_spec(tp=2)
    plan = LaunchPlan(("python",))
    launcher.start("r", replica.resource, plan)
    launcher.start("r", replica.resource, plan)
    pg_api.placement_group.assert_called_once_with(
        [{"CPU": replica.resource.slot.num_cpus, "NPU": 2, "rollout": 0.001}],
        strategy="STRICT_PACK",
    )
    assert launcher.alive("r")
    fake_ray.remote.assert_not_called()
    launcher.stop("r")
    launcher.stop("r")
    assert not launcher.groups and not launcher.pending


def test_actor_uses_reserved_bundle_and_cleanup_precedes_pg_removal(fake_ray, pg_api):
    launcher = RayPGProcessLauncher(resource_label="rollout")
    replica, _ = replica_and_spec(tp=2)
    launcher.start("r", replica.resource, LaunchPlan(("python",)))
    fake_ray.wait.side_effect = lambda refs, **kwargs: (refs, [])
    fake_ray.get.side_effect = [None, ("0", "1"), "http://10.0.0.8:8000"]
    assert launcher.started("r")
    assert replica.resource.allocation.endpoint == "http://10.0.0.8:8000"
    options = fake_ray.remote.return_value.options.call_args.kwargs
    assert options["scheduling_strategy"]["placement_group_bundle_index"] == 0
    fake_ray.get.side_effect = TimeoutError("cleanup not confirmed")
    with pytest.raises(TimeoutError):
        launcher.stop("r")
    pg_api.remove_placement_group.assert_not_called()
    fake_ray.get.side_effect = None
    fake_ray.get.return_value = ("0", "1")
    pg_api.placement_group_table.return_value = {"state": "CREATED"}
    with pytest.raises(RuntimeError, match="removal pending"):
        launcher.stop("r")
    assert "r" in launcher.groups
    pg_api.placement_group_table.return_value = {"state": "REMOVED"}
    launcher.stop("r")
    assert not launcher.groups


@pytest.mark.parametrize("strategy", ["fsdp2", "megatron"])
def test_pg_entry_accepts_future_capacity_and_pins_training(tmp_path, strategy):
    config, nodes = native_config(tmp_path, accelerator="npu", strategy=strategy)
    nodes[0]["Resources"]["NPU"] = 4  # All currently existing devices are for training.
    spec, child = build_launch(
        config,
        nodes,
        "10.0.0.1:6379",
        "10.0.0.1",
        NativeOptions(resource_backend="ray_pg", pg_resource_label="rollout", max_replicas=8),
    )
    core = spec.core_config()
    assert len(core["slots"]) == 8
    assert all(s["placement_group"] and not s["node_id"] for s in core["slots"])
    assert child.trainer.placement_group_node_ids.global_pool == ["n0"]
    validate_verl_config(spec, spec.ray_address, [], core, config=child)
