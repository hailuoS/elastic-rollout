import threading
from types import SimpleNamespace as NS

import pytest

from elastic_rollout.collective_backend import CollectiveBackend
from elastic_rollout.engine.vllm import StandardVLLMControl
from elastic_rollout.engine.weight_transfer_worker import WeightTransferWorker
from elastic_rollout.hccl_sender import HCCLSender
from elastic_rollout.runtime import PolicyConflict


def test_receiver_failure_quarantines_the_slot_and_blocks_sender_ack(elastic):
    from test_mvp4 import grow, update

    c, m, driver, _ = elastic
    (candidate,) = grow(c, m, 3)
    update(m, 1)
    m.set_desired(2)
    c.reconcile()
    c.reconcile()
    (request,) = m.transfer.maintenance()["requests"]

    def fail(*args, **kwargs):
        raise RuntimeError("receiver native destroy failed")

    driver.weight_transfer_control.destroy_weight_transfer = fail
    with pytest.raises(RuntimeError, match="receiver native"):
        m.transfer.maintenance({"destroy_receivers": request})
    with pytest.raises(PolicyConflict, match="receiver destroy"):
        m.transfer.maintenance({"acknowledgements": [request]})
    assert m.transfer.poisoned and candidate.resource and m.resources.allocated == 3
    assert candidate.replica_id not in driver.stopped


@pytest.mark.parametrize("backend", ["hccl", "nccl"])
def test_retirement_enters_both_native_sides_before_ack(monkeypatch, backend):
    owner = threading.get_ident()
    barrier = threading.Barrier(2, timeout=2)
    calls = []

    def destroy(group):
        calls.append((group, threading.get_ident()))
        barrier.wait()

    monkeypatch.setattr(CollectiveBackend, "destroy", lambda self, group: destroy(group))
    sender = HCCLSender(
        1024,
        True,
        endpoint="http://unused",
        deployment_id="test",
        backend=backend,
        master_address="trainer",
    )
    sender.prepare([{"transport_key": "dynamic"}])
    child = sender._children["dynamic"]
    child.transport_id, child.group = "t", "sender"
    receiver = WeightTransferWorker()
    receiver.weight_transfer_engine = NS(model_update_group="receiver")

    def control(body, **kwargs):
        token = body["destroy_receivers"]["retirement_id"]
        assert receiver.elastic_destroy_weight_transfer(backend, token) == {"retirement_id": token}
        return {"receiver_destroyed": token}

    sender.control.transport_maintenance = control
    request = dict(transport_key="dynamic", transport_id="t", retirement_id="r")
    assert sender.retire_transports([request]) == {
        "acknowledgements": [{"transport_key": "dynamic", "retirement_id": "r"}]
    }
    assert ("sender", owner) in calls and len(calls) == 2
    assert receiver.weight_transfer_engine.model_update_group is None
    assert receiver.elastic_destroy_weight_transfer(backend, "r") == {"retirement_id": "r"}
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="another identity"):
        receiver.elastic_destroy_weight_transfer(backend, "stale")


@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"retirement_id": "r"}],
        [None, None],
        [{"retirement_id": "r"}, {"retirement_id": "old"}],
    ],
)
def test_receiver_proof_must_cover_every_tp_rank(monkeypatch, results):
    monkeypatch.setattr(
        "elastic_rollout.engine.vllm.request_json", lambda *a, **k: {"results": results}
    )
    with pytest.raises(RuntimeError, match="every worker"):
        StandardVLLMControl().destroy_weight_transfer("http://unused", "nccl", "r", 2)
