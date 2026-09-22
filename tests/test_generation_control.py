"""Deployment pause gates, physical control, and snapshot transaction ordering."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from test_policy_updates import prepare
from test_runtime import ready

from elastic_rollout.api import ReplicaState as S
from elastic_rollout.client import ControlError
from elastic_rollout.runtime import PolicyConflict
from elastic_rollout.traffic import NoReadyReplica


@pytest.mark.parametrize("first", ["generation", "policy_update"])
def test_composable_pause_reasons(runtime, first):
    traffic = ready(runtime)
    other = "policy_update" if first == "generation" else "generation"
    traffic.pause_routes(first)
    traffic.pause_routes(other)
    traffic.pause_routes(first)
    assert traffic.route_pause_reasons == ("generation", "policy_update")
    traffic.resume_routes(first)
    traffic.resume_routes(first)
    assert traffic.route_pause_reasons == (other,)
    with pytest.raises(NoReadyReplica, match=other):
        traffic.acquire("blocked")
    traffic.resume_routes(other)
    assert traffic.accepting_routes


def test_http_pause_resume_idempotent_and_reject_scale(http_runtime, monkeypatch):
    client, runtime = http_runtime
    driver = runtime[2]
    lease = client.acquire("running")
    invoked = threading.Event()
    original = driver.pause_generation

    def pause(replica, **kwargs):
        original(replica, **kwargs)
        invoked.set()

    monkeypatch.setattr(driver, "pause_generation", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.pause_generation)
        assert invoked.wait(5)
        assert not pending.done()
        # Planned pause preserves valid leases until the partial response is consumed.
        assert client.renew(lease.lease_id).lease_id == lease.lease_id
        assert client.release(lease.lease_id)
        assert pending.result(5)["inflight"] == 0
    pause_calls = len(driver.pause_calls)
    for _ in range(2):
        status = client.pause_generation()
        assert status["generation_paused"]
        assert status["route_pause_reasons"] == ["generation"]
    assert len(driver.pause_calls) == pause_calls >= 2
    with pytest.raises(ControlError) as error:
        client.acquire("blocked")
    assert error.value.status == 503
    assert client.status()["inflight"] == 0
    with pytest.raises(ControlError) as error:
        client.set_desired_replicas(4)
    assert error.value.status == 409
    for _ in range(2):
        status = client.resume_generation()
        assert not status["generation_paused"]
        assert status["route_pause_reasons"] == []
    assert len(driver.resume_calls) == 2
    assert client.status()["inflight"] == 0


@pytest.mark.parametrize("action", ["pause_generation", "resume_generation"])
def test_partial_control_failure_stays_closed_and_can_recover(http_runtime, monkeypatch, action):
    client, runtime = http_runtime
    manager, driver = runtime[1:3]
    if action == "resume_generation":
        client.pause_generation()
    original = getattr(driver, action)
    calls = []

    def fail_second(replica, **kwargs):
        original(replica)  # A lost response is ambiguous: physical control already happened.
        calls.append(replica.replica_id)
        if len(calls) == 2:
            raise TimeoutError("lost control response")

    monkeypatch.setattr(driver, action, fail_second)
    with pytest.raises(ControlError) as error:
        getattr(client, action)()
    assert error.value.status == 500
    status = client.status()
    assert status["route_pause_reasons"] == ["generation"]
    assert "lost control response" in status["last_error"]
    with pytest.raises(PolicyConflict):
        manager.set_desired(4)
    # Pause after failed resume must not be skipped by the previous paused flag.
    monkeypatch.setattr(driver, action, original)
    client.pause_generation()
    assert len(driver.paused) == 2
    client.resume_generation()
    assert not driver.paused and client.status()["accepting_routes"]


def test_resume_recovers_failed_first_pause(runtime, monkeypatch):
    ready(runtime)
    manager, driver = runtime[1:3]
    original = driver.pause_generation

    def fail(replica, **kwargs):
        original(replica)
        raise TimeoutError("pause response lost")

    monkeypatch.setattr(driver, "pause_generation", fail)
    with pytest.raises(RuntimeError):
        manager.pause_generation()
    assert not manager.generation_paused
    manager.resume_generation()
    assert not driver.paused and manager.traffic.accepting_routes


@pytest.mark.parametrize("commit_first", [False, True])
def test_prepared_resume_and_commit_remove_only_their_reason(runtime, commit_first):
    traffic = ready(runtime)
    controller, manager, driver, _ = runtime
    manager.pause_generation()
    update_id = prepare(runtime)  # Begin is explicitly allowed during generation pause.
    assert manager.generation_paused
    assert not driver.paused  # Fresh vLLM processes can warm up behind the route gate.
    assert traffic.route_pause_reasons == ("generation", "policy_update")

    def commit():
        return controller.commit_policy_update("test", update_id, 1)

    first, second = (
        (commit, manager.resume_generation)
        if commit_first
        else (
            manager.resume_generation,
            commit,
        )
    )
    first()
    assert traffic.route_pause_reasons == (("generation",) if commit_first else ("policy_update",))
    with pytest.raises(NoReadyReplica):
        traffic.acquire("too-early")
    second()
    lease = traffic.acquire("committed")
    assert lease.serving_policy_version == 1
    traffic.release(lease.lease_id)
    controller.set_desired_replicas("test", 4)
    controller.reconcile()
    controller.reconcile()
    assert manager.status()["ready_replicas"] == 4
    controller.set_desired_replicas("test", 2)
    controller.reconcile()
    runtime[3].now += manager.spec.drain_grace_period + 1
    controller.reconcile()
    assert manager.status()["total_replicas"] == manager.status()["ready_replicas"] == 2


def test_failed_policy_update_stays_closed_after_generation_resume(runtime):
    traffic = ready(runtime)
    controller, manager, driver, _ = runtime
    manager.pause_generation()
    controller.begin_policy_update("test", "/shared/candidate")
    driver.fail_stop = True
    controller.reconcile()
    assert manager.policy_update.state == "FAILED"
    driver.fail_stop = False
    manager.resume_generation()
    assert traffic.route_pause_reasons == ("policy_update",)
    assert not traffic.accepting_routes


def test_pause_allows_crash_replacement_and_resume_uses_current_targets(runtime):
    traffic = ready(runtime)
    controller, manager, driver, _ = runtime
    victim = next(iter(traffic.replicas))
    manager.pause_generation()
    driver.processes.remove(victim)
    controller.reconcile()
    controller.reconcile()
    targets = {r.replica_id for r in traffic.replicas.values() if r.state == S.READY}
    assert len(targets) == 2 and victim not in targets
    assert not traffic.accepting_routes
    manager.resume_generation()
    assert set(driver.resume_calls) == targets


def test_pause_includes_draining_replicas(runtime):
    traffic = ready(runtime)
    controller, manager, driver, _ = runtime
    controller.set_desired_replicas("test", 1)
    controller.reconcile()
    assert any(r.state == S.DRAINING for r in traffic.replicas.values())
    manager.pause_generation()
    assert set(driver.pause_calls) == set(traffic.replicas)


@pytest.mark.parametrize("action", ["pause_generation", "resume_generation"])
def test_control_serializes_but_leases_and_status_remain_live(runtime, monkeypatch, action):
    traffic = ready(runtime)
    controller, manager, driver, _ = runtime
    if action == "resume_generation":
        manager.pause_generation()
    else:
        lease = traffic.acquire("running")
    entered, release = threading.Event(), threading.Event()
    original = getattr(driver, action)

    def blocked(replica, **kwargs):
        entered.set()
        assert release.wait(5)
        original(replica)

    operation = MagicMock(side_effect=blocked)
    monkeypatch.setattr(driver, action, operation)
    with ThreadPoolExecutor(max_workers=5) as pool:
        first = pool.submit(getattr(manager, action))
        try:
            assert entered.wait(5)
            duplicate = pool.submit(getattr(manager, action))
            reconcile = pool.submit(controller.reconcile)
            # Separate threads make accidentally holding traffic.lock fail promptly.
            assert not pool.submit(manager.status).result(2)["accepting_routes"]
            if action == "pause_generation":
                assert pool.submit(traffic.release, lease.lease_id).result(2)
            assert not duplicate.done() and not reconcile.done()
        finally:
            release.set()
        first.result(5)
        duplicate.result(5)
        reconcile.result(5)
    assert operation.call_count == 2  # Once per physical target, not per caller.


def test_shutdown_reason_survives_generation_resume(runtime):
    traffic = ready(runtime)
    manager = runtime[1]
    manager.pause_generation()
    traffic.pause_routes("shutdown")
    manager.resume_generation()
    assert traffic.route_pause_reasons == ("shutdown",)
