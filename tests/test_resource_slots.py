import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.engine.vllm import VLLMServiceDriver
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_positive_tp_and_matching_slot(tp):
    assert ReplicaSpec("model", tensor_parallel_size=tp).tensor_parallel_size == tp
    assert ResourceSlot("s", "node", "host", tp).num_devices == tp


@pytest.mark.parametrize("tp,pp", [(0, 1), (-1, 1), (True, 1), (2.5, 1), (2, 2), (2, True)])
def test_invalid_replica_shape(tp, pp):
    with pytest.raises(ValueError):
        ReplicaSpec("model", tensor_parallel_size=tp, pipeline_parallel_size=pp)


@pytest.mark.parametrize("count", [0, -1, True, 2.5, "2"])
def test_invalid_device_count_rejected(count):
    with pytest.raises(ValueError, match="num_devices must be a positive integer"):
        ResourceSlot("s", "node", "host", count)


@pytest.mark.parametrize("same_node,same_host", [(True, True), (True, False), (False, True)])
def test_same_node_port_conflict(same_node, same_host):
    a = ResourceSlot("a", "node", "host", 2)
    b = ResourceSlot(
        "b", "node" if same_node else "alias", "host" if same_host else "alias", 2, port=8000
    )
    with pytest.raises(ValueError, match="node .*port 8000"):
        StaticResourceProvider([a, b])


def test_different_nodes_can_reuse_port():
    slots = [ResourceSlot(str(i), f"node{i}", f"host{i}", 2) for i in range(2)]
    provider = StaticResourceProvider(slots)
    assert provider.acquire("r0").slot != provider.acquire("r1").slot


def test_tp_slot_mismatch_fails_before_launch_and_releases_allocation():
    launcher = MagicMock()
    resources = StaticResourceProvider([ResourceSlot("s", "node", "host", 4)])
    controller = Controller(resources, VLLMServiceDriver(launcher))
    controller.ensure_deployment(
        DeploymentSpec("d", ReplicaSpec("model", tensor_parallel_size=2), 1)
    )
    controller.reconcile()
    launcher.start.assert_not_called()
    assert resources.allocated == 0
    assert controller.deployments["d"].last_error == (
        "Replica TP=2 requires a ResourceSlot with exactly 2 devices, got 4."
    )


def test_elastic_example_has_four_capacity_slots():
    path = Path(__file__).resolve().parents[1] / "examples/runtime.elastic.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    slots = [ResourceSlot(**slot) for slot in config["slots"]]
    spec = DeploymentSpec.from_dict(config["deployments"][0])
    provider = StaticResourceProvider(slots)
    leases = [provider.acquire(str(i)) for i in range(4)]
    assert len({lease.slot.node_id for lease in leases}) == 4
    assert [lease.slot.num_devices for lease in leases] == [2, 2, 2, 2]
    assert len({(lease.slot.host, lease.slot.port) for lease in leases}) == 4
    assert spec.replica.tensor_parallel_size == 2 and spec.desired_replicas == 2
