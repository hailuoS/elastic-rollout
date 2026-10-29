from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from elastic_rollout.api import ReplicaSpec
from elastic_rollout.api import ReplicaState as S
from elastic_rollout.runtime import Reconciler
from elastic_rollout.traffic import NoReadyReplica
from tests.helpers import ready


def test_only_ready_receives_routes(runtime):
    controller, manager, _, _ = runtime
    controller.reconcile()
    with pytest.raises(NoReadyReplica):
        manager.traffic.acquire("a")
    controller.reconcile()
    assert manager.traffic.acquire("a").serving_policy_version == 0


def test_sticky_least_inflight_duplicate_release_and_ttl(runtime):
    traffic = ready(runtime)
    a = traffic.acquire("a")
    b = traffic.acquire("b")
    assert a.replica_id != b.replica_id
    assert traffic.release(a.lease_id)
    assert not traffic.release(a.lease_id)
    assert traffic.acquire("a").replica_id == a.replica_id
    runtime[3].now += 21
    traffic.expire()
    assert sum(traffic.inflight.values()) == 0
    assert not traffic.release(b.lease_id)


def test_scale_2_4_2_and_drain_preserves_inflight(runtime):
    controller, manager, driver, clock = runtime
    traffic = ready(runtime)
    controller.set_desired_replicas("test", 4)
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 2
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 4
    leases = [traffic.acquire(str(i)) for i in range(4)]
    controller.set_desired_replicas("test", 2)
    controller.reconcile()
    draining = [r for r in traffic.replicas.values() if r.state == S.DRAINING]
    victim = draining[0]
    old = next(lease for lease in leases if lease.replica_id == victim.replica_id)
    sticky = traffic.acquire(old.routing_key)
    assert sticky.replica_id == victim.replica_id
    assert traffic.acquire("new").replica_id not in {r.replica_id for r in draining}
    clock.now += 11
    controller.reconcile()
    assert victim.replica_id not in driver.stopped
    assert traffic.acquire(old.routing_key).replica_id != victim.replica_id
    for lease in list(traffic.leases.values()):
        traffic.release(lease.lease_id)
    controller.reconcile()
    assert manager.status()["total_replicas"] == 2
    assert controller.resources.allocated == 2
    controller.set_desired_replicas("test", 0)
    controller.reconcile()
    clock.now += 11
    controller.reconcile()
    assert controller.resources.allocated == 0
    assert not driver.processes


def test_expired_lease_does_not_kill_busy_server(runtime):
    controller, manager, driver, clock = runtime
    traffic = ready(runtime)
    lease = traffic.acquire("busy")
    driver.busy.add(lease.replica_id)
    controller.set_desired_replicas("test", 0)
    controller.reconcile()
    clock.now += 30
    controller.reconcile()
    assert lease.replica_id in driver.processes
    assert not traffic.leases
    driver.busy.clear()
    controller.reconcile()
    assert controller.resources.allocated == 0


@pytest.mark.parametrize("failure", ["launch", "health", "crash"])
def test_failure_releases_resources_and_replaces(runtime, failure):
    controller, manager, driver, clock = runtime
    if failure == "launch":
        driver.fail_launch = True
        controller.reconcile()
        driver.fail_launch = False
    elif failure == "health":
        driver.healthy = False
        controller.reconcile()
        clock.now += 601
        controller.reconcile()
        driver.healthy = True
    else:
        ready(runtime)
        driver.processes.clear()
    controller.reconcile()
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 2
    assert controller.resources.allocated == 2
    assert any(r.state == S.FAILED for r in manager.traffic.replicas.values())


def test_stop_failure_quarantines_resources(runtime):
    controller, manager, driver, clock = runtime
    ready(runtime)
    controller.set_desired_replicas("test", 0)
    controller.reconcile()
    driver.fail_stop = True
    clock.now += 11
    controller.reconcile()
    assert controller.resources.allocated == 2
    driver.fail_stop = False
    controller.reconcile()
    assert controller.resources.allocated == 0


def test_concurrent_routes_and_idempotent_desired_state(runtime):
    controller, manager, _, _ = runtime
    traffic = ready(runtime)
    with ThreadPoolExecutor(max_workers=16) as executor:
        leases = list(executor.map(traffic.acquire, ["same"] * 100))
    assert len({x.replica_id for x in leases}) == 1
    assert sum(traffic.inflight.values()) == 100
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(traffic.release, [x.lease_id for x in leases] * 2))
    assert sum(traffic.inflight.values()) == 0
    controller.set_desired_replicas("test", 4)
    controller.ensure_deployment(manager.spec)
    assert manager.desired == 4
    for _ in range(5):
        controller.reconcile()
    assert len(manager.traffic.replicas) == 4


def test_lease_renewal_and_unknown_lease(runtime):
    traffic = ready(runtime)
    lease = traffic.acquire("a")
    runtime[3].now += 10
    renewed = traffic.renew(lease.lease_id)
    assert renewed.expires_at > lease.expires_at
    assert sum(traffic.inflight.values()) == 1
    runtime[3].now += 21
    with pytest.raises(KeyError):
        traffic.renew(lease.lease_id)


def test_scale_reversal_converges_without_overallocation(runtime):
    controller, manager, _, clock = runtime
    ready(runtime)
    controller.set_desired_replicas("test", 0)
    controller.reconcile()
    controller.set_desired_replicas("test", 4)
    controller.reconcile()
    assert controller.resources.allocated == 2
    clock.now += 11
    controller.reconcile()
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 4


def test_validation_and_capacity_exhaustion(runtime):
    with pytest.raises(ValueError):
        ReplicaSpec("model", tensor_parallel_size=0)
    with pytest.raises(ValueError):
        runtime[0].set_desired_replicas("test", True)
    runtime[0].set_desired_replicas("test", 5)
    for _ in range(3):
        runtime[0].reconcile()
    assert runtime[1].status()["ready_replicas"] == 4
    assert "no free" in runtime[1].last_error


def test_startup_waits_for_warmup_before_routing(runtime, monkeypatch):
    controller, manager, driver, _ = runtime
    monkeypatch.setattr(driver, "warmup", lambda *args: False)
    controller.reconcile()
    controller.reconcile()
    assert {r.state for r in manager.traffic.replicas.values()} == {S.WARMING}
    with pytest.raises(NoReadyReplica):
        manager.traffic.acquire("waiting-for-warmup")
    monkeypatch.setattr(driver, "warmup", lambda *args: True)
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 2


def test_serving_health_recovers_and_replaces_after_three_consecutive_failures(runtime):
    controller, manager, driver, _ = runtime
    traffic = ready(runtime)
    original = list(traffic.replicas.values())
    driver.healthy = False
    controller.reconcile()
    controller.reconcile()
    assert all(r.state == S.READY and r.health_failures == 2 for r in original)
    driver.healthy = True
    controller.reconcile()
    assert all(r.health_failures == 0 for r in original)
    driver.healthy = False
    for _ in range(3):
        controller.reconcile()
    assert all(r.state == S.FAILED and r.resource is None for r in original)
    driver.healthy = True
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 2


def test_reconciler_logs_exception_and_continues(caplog):
    controller = Mock()
    recovered = Event()
    error = RuntimeError("transient reconcile failure")

    def reconcile():
        if controller.reconcile.call_count == 1:
            raise error
        recovered.set()

    controller.reconcile.side_effect = reconcile
    reconciler = Reconciler(controller, interval=0.001)
    reconciler.start()
    try:
        assert recovered.wait(timeout=2)
    finally:
        reconciler.stop()
    records = [r for r in caplog.records if r.name == "elastic_rollout.runtime"]
    assert len(records) == 1
    assert records[0].exc_info[1] is error
    assert "reconcile failed" in records[0].getMessage()
