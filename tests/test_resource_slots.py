import pytest

from elastic_rollout.api import ReplicaSpec, ResourceSlot
from elastic_rollout.resource import StaticResourceProvider


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
