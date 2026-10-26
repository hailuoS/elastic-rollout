import pytest

from elastic_rollout.api import ReplicaState as S
from tests.helpers import ready


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_replica_invalidates_only_its_leases_before_cleanup(runtime, cleanup_fails):
    controller, manager, driver, clock = runtime
    traffic = ready(runtime)
    dead = [traffic.acquire("victim") for _ in range(5)]
    other = traffic.acquire("survivor")
    victim = traffic.replicas[dead[0].replica_id]
    assert other.replica_id != victim.replica_id
    driver.processes.remove(victim.replica_id)
    driver.fail_stop = cleanup_fails
    controller.reconcile()
    assert victim.state == S.FAILED
    assert traffic.inflight[victim.replica_id] == 0
    assert traffic.inflight[other.replica_id] == 1
    assert set(traffic.leases) == {other.lease_id}
    assert "victim" not in traffic.sticky
    assert traffic.sticky["survivor"] == other.replica_id
    assert traffic.invalidate_replica(victim.replica_id) == 0
    for lease in dead:
        with pytest.raises(KeyError):
            traffic.renew(lease.lease_id)
        assert not traffic.release(lease.lease_id)
    clock.now += 1
    assert traffic.renew(other.lease_id).expires_at > other.expires_at
    driver.fail_stop = False
    controller.reconcile()
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 2
    replacement = next(
        r
        for r in traffic.replicas.values()
        if r.replica_id not in (victim.replica_id, other.replica_id)
    )
    assert replacement.serving_policy_version == manager.spec.replica.policy_version
    assert traffic.inflight[victim.replica_id] == 0


def test_draining_replica_cannot_invalidate_and_keeps_renewing(runtime):
    controller, manager, driver, clock = runtime
    traffic = ready(runtime)
    lease = traffic.acquire("drain")
    controller.set_desired_replicas("test", 0)
    controller.reconcile()
    assert traffic.replicas[lease.replica_id].state == S.DRAINING
    with pytest.raises(ValueError, match="FAILED"):
        traffic.invalidate_replica(lease.replica_id)
    clock.now += manager.spec.drain_grace_period + 1
    controller.reconcile()
    assert traffic.renew(lease.lease_id).expires_at > lease.expires_at
    assert lease.replica_id not in driver.stopped
    assert traffic.release(lease.lease_id)
    controller.reconcile()
    assert lease.replica_id in driver.stopped
