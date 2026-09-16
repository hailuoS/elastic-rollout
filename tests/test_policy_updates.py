from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from test_runtime import ready

from elastic_rollout.api import PolicyUpdateState as U
from elastic_rollout.api import ReplicaState as S
from elastic_rollout.runtime import PolicyConflict
from elastic_rollout.traffic import NoReadyReplica

SNAPSHOT = "/shared/policy-1"


def prepare(runtime):
    controller, manager, _, _ = runtime
    update = controller.begin_policy_update("test", SNAPSHOT)
    for _ in range(8):
        controller.reconcile()
        if manager.policy_update.state == U.PREPARED:
            return update["update_id"]
    pytest.fail(f"update did not prepare: {manager.status()}")


def test_bootstrap_and_prepare_commit_keeps_lease_and_versions(runtime):
    controller, manager, driver, _ = runtime
    traffic = ready(runtime)
    assert all(path == "model" and version == 0 for _, path, version in driver.starts)
    resources = {r.replica_id: r.resource for r in traffic.replicas.values()}
    release = manager.resources.release
    manager.resources.release = MagicMock(wraps=release)
    update_id = prepare(runtime)
    manager.resources.release.assert_not_called()
    assert manager.resources.allocated == 2
    assert {r.replica_id: r.resource for r in traffic.replicas.values()} == resources
    assert all(r.loaded_model_path == SNAPSHOT for r in traffic.replicas.values())
    assert all(r.serving_policy_version == 0 for r in traffic.replicas.values())
    assert driver.starts[-2:][0][1] == SNAPSHOT
    with pytest.raises(NoReadyReplica, match="paused"):
        traffic.acquire("candidate")
    assert manager.status()["policy"] == {"version": 0, "model_path": "model"}
    committed = controller.commit_policy_update("test", update_id, 1)
    assert committed["state"] == U.COMMITTED
    assert committed["history"] == [
        s.value
        for s in (U.REQUESTED, U.QUIESCING, U.RESTARTING, U.VERIFYING, U.PREPARED, U.COMMITTED)
    ]
    assert controller.commit_policy_update("test", update_id, 1) == committed
    assert all(r.serving_policy_version == 1 for r in traffic.replicas.values())
    assert traffic.acquire("committed").serving_policy_version == 1
    assert manager.status()["policy"] == {"version": 1, "model_path": SNAPSHOT}
    assert manager.spec.replica.model == "model" and manager.spec.replica.policy_version == 0


def test_quiescing_allows_existing_lease_renew_release_and_waits_for_idle(runtime):
    controller, manager, driver, clock = runtime
    traffic = ready(runtime)
    lease = traffic.acquire("sticky")
    controller.begin_policy_update("test", SNAPSHOT)
    for key in ("new", "sticky"):
        with pytest.raises(NoReadyReplica):
            traffic.acquire(key)
    controller.reconcile()
    assert manager.policy_update.state == U.QUIESCING
    clock.now += 1
    assert traffic.renew(lease.lease_id).expires_at > lease.expires_at
    driver.busy.add(lease.replica_id)
    assert traffic.release(lease.lease_id)
    controller.reconcile()
    assert not driver.stopped
    driver.busy.clear()
    controller.reconcile()
    assert len(driver.stopped) == 2


def test_lease_expiry_is_not_backend_idle(runtime):
    controller, manager, driver, clock = runtime
    traffic = ready(runtime)
    lease = traffic.acquire("running")
    driver.busy.add(lease.replica_id)
    controller.begin_policy_update("test", SNAPSHOT)
    clock.now += manager.spec.lease_ttl + 1
    controller.reconcile()
    assert not traffic.leases
    assert manager.policy_update.state == U.QUIESCING
    assert not driver.stopped


def test_scale_and_crash_replacement_use_latest_committed_snapshot(runtime):
    controller, manager, driver, _ = runtime
    traffic = ready(runtime)
    update_id = prepare(runtime)
    controller.commit_policy_update("test", update_id, 3)
    controller.set_desired_replicas("test", 4)
    controller.reconcile()
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 4
    assert all(path == SNAPSHOT and version == 3 for _, path, version in driver.starts[-2:])
    crashed = next(r for r in traffic.replicas.values() if r.state == S.READY)
    driver.processes.remove(crashed.replica_id)
    controller.reconcile()
    controller.reconcile()
    assert driver.starts[-1][1:] == (SNAPSHOT, 3)
    assert manager.status()["ready_replicas"] == 4


def test_candidate_launch_failure_replaces_from_candidate_without_routes(runtime):
    controller, manager, driver, _ = runtime
    traffic = ready(runtime)
    controller.begin_policy_update("test", SNAPSHOT)
    driver.fail_launch = True
    controller.reconcile()
    assert manager.policy_update.state == U.RESTARTING
    driver.fail_launch = False
    for _ in range(5):
        controller.reconcile()
        with pytest.raises(NoReadyReplica):
            traffic.acquire("blocked")
    assert manager.policy_update.state == U.PREPARED
    assert all(path == SNAPSHOT and version == 0 for _, path, version in driver.starts[2:])
    assert any(r.state == S.FAILED for r in traffic.replicas.values())
    assert manager.committed_policy.version == 0


def test_prepared_failure_requires_reverification(runtime):
    controller, manager, driver, _ = runtime
    ready(runtime)
    update_id = prepare(runtime)
    driver.processes.remove(next(iter(driver.processes)))
    controller.reconcile()
    with pytest.raises(PolicyConflict):
        controller.commit_policy_update("test", update_id, 1)
    for _ in range(4):
        controller.reconcile()
    assert manager.policy_update.state == U.PREPARED
    controller.commit_policy_update("test", update_id, 1)


def test_unrecoverable_stop_failure_is_closed_without_releasing_resources(runtime):
    controller, manager, driver, _ = runtime
    traffic = ready(runtime)
    update = controller.begin_policy_update("test", SNAPSHOT)
    driver.fail_stop = True
    controller.reconcile()
    assert manager.policy_update.state == U.FAILED
    assert manager.status()["policy_update"]["error"] == "stop failed"
    assert manager.resources.allocated == 2
    assert manager.committed_policy.version == 0
    driver.fail_stop = False
    controller.reconcile()
    with pytest.raises(NoReadyReplica):
        traffic.acquire("still-closed")
    with pytest.raises(PolicyConflict):
        controller.commit_policy_update("test", update["update_id"], 1)


def test_conflicts_and_input_validation(runtime):
    controller, manager, _, _ = runtime
    controller.reconcile()
    with pytest.raises(PolicyConflict):
        controller.begin_policy_update("test", SNAPSHOT)
    controller.reconcile()
    for path in (None, "", " ", 1):
        with pytest.raises(ValueError):
            controller.begin_policy_update("test", path)
    update = controller.begin_policy_update("test", SNAPSHOT)
    with pytest.raises(PolicyConflict):
        controller.set_desired_replicas("test", 4)
    with pytest.raises(PolicyConflict):
        controller.begin_policy_update("test", "/shared/second")
    with pytest.raises(PolicyConflict):
        controller.commit_policy_update("test", update["update_id"], 1)
    for _ in range(4):
        controller.reconcile()
    for version in (-1, 0, True, 1.5, "1"):
        with pytest.raises(ValueError):
            controller.commit_policy_update("test", update["update_id"], version)
    assert not manager.traffic.accepting_routes


def test_control_mutations_fail_fast_during_slow_reconcile(runtime):
    controller, manager, _, _ = runtime
    ready(runtime)
    with manager.mutation_lock, ThreadPoolExecutor() as pool:
        future = pool.submit(controller.begin_policy_update, "test", SNAPSHOT)
        with pytest.raises(PolicyConflict):
            future.result(timeout=1)
        # A normal MVP-0 scale request only changes desired state and remains
        # available while health/process operations hold the lifecycle lock.
        assert controller.set_desired_replicas("test", 4)["desired_replicas"] == 4


def test_atomic_commit_never_routes_old_version_on_candidate(runtime):
    controller, manager, _, _ = runtime
    traffic = ready(runtime)
    update_id = prepare(runtime)

    def acquire_or_paused(_):
        try:
            lease = traffic.acquire("racing")
            traffic.release(lease.lease_id)
            return lease.serving_policy_version
        except NoReadyReplica:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        before = [pool.submit(acquire_or_paused, i) for i in range(30)]
        controller.commit_policy_update("test", update_id, 1)
        after = [pool.submit(acquire_or_paused, i) for i in range(30)]
        assert {f.result() for f in before + after} <= {None, 1}


def test_shutdown_can_drain_an_active_update(runtime):
    controller, manager, _, clock = runtime
    ready(runtime)
    controller.begin_policy_update("test", SNAPSHOT)
    # Drive grace using the fake clock while close owns desired=0.
    manager.spec = replace(manager.spec, drain_grace_period=0)
    controller.close(timeout=2)
    assert manager.resources.allocated == 0
    assert manager.policy_update.state == U.FAILED
