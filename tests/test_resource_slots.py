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
    assert len(ResourceSlot("s", "node", "host", tuple(map(str, range(tp)))).device_ids) == tp


@pytest.mark.parametrize("tp,pp", [(0, 1), (-1, 1), (True, 1), (2.5, 1), (2, 2), (2, True)])
def test_invalid_replica_shape(tp, pp):
    with pytest.raises(ValueError):
        ReplicaSpec("model", tensor_parallel_size=tp, pipeline_parallel_size=pp)


@pytest.mark.parametrize("devices", [(), ("0", "0"), (0, "0")])
def test_empty_or_duplicate_devices_rejected(devices):
    with pytest.raises(ValueError, match="nonempty.*distinct"):
        ResourceSlot("s", "node", "host", devices)


@pytest.mark.parametrize("same_node,same_host", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize(
    "devices,port,error", [(("1", "2"), 8001, "devices.*1"), (("2", "3"), 8000, "port 8000")]
)
def test_same_node_conflicts_identify_device_or_port(same_node, same_host, devices, port, error):
    a = ResourceSlot("a", "node", "host", ("0", "1"))
    b = ResourceSlot(
        "b", "node" if same_node else "alias", "host" if same_host else "alias", devices, port=port
    )
    with pytest.raises(ValueError, match=f"node .*{error}"):
        StaticResourceProvider([a, b])


def test_different_nodes_can_reuse_devices_and_port():
    slots = [ResourceSlot(str(i), f"node{i}", f"host{i}", ("0", "1")) for i in range(2)]
    provider = StaticResourceProvider(slots)
    assert provider.acquire("r0").slot != provider.acquire("r1").slot


def test_tp_slot_mismatch_fails_before_launch_and_releases_allocation():
    launcher = MagicMock()
    resources = StaticResourceProvider([ResourceSlot("s", "node", "host", ("0", "1", "2", "3"))])
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


def test_single_node_example_has_four_disjoint_tp2_slots():
    path = Path(__file__).resolve().parents[1] / "examples/runtime.single-node-tp2.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    slots = [ResourceSlot(**slot) for slot in config["slots"]]
    spec = DeploymentSpec.from_dict(config["deployments"][0])
    provider = StaticResourceProvider(slots)
    leases = [provider.acquire(str(i)) for i in range(4)]
    assert len({lease.slot.node_id for lease in leases}) == 1
    assert [lease.slot.device_ids for lease in leases] == [
        ("0", "1"),
        ("2", "3"),
        ("4", "5"),
        ("6", "7"),
    ]
    assert spec.replica.tensor_parallel_size == 2 and spec.desired_replicas == 2
