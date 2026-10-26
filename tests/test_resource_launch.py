import json
from pathlib import Path
from unittest.mock import MagicMock

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.engine.vllm import VLLMServiceDriver
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller


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


def test_single_node_example_has_four_capacity_slots():
    path = Path(__file__).resolve().parents[1] / "examples/runtime.single-node-tp2.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    slots = [ResourceSlot(**slot) for slot in config["slots"]]
    spec = DeploymentSpec.from_dict(config["deployments"][0])
    provider = StaticResourceProvider(slots)
    leases = [provider.acquire(str(i)) for i in range(4)]
    assert len({lease.slot.node_id for lease in leases}) == 1
    assert [lease.slot.num_devices for lease in leases] == [2, 2, 2, 2]
    assert len({lease.slot.port for lease in leases}) == 4
    assert spec.replica.tensor_parallel_size == 2 and spec.desired_replicas == 2
