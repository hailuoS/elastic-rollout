from concurrent.futures import ThreadPoolExecutor

import pytest

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.api import ReplicaState as S
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller
from elastic_rollout.traffic import NoReadyReplica


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class FakeDriver:
    def __init__(self):
        self.processes = set()
        self.stopped = []
        self.healthy = True
        self.busy = set()
        self.fail_launch = False
        self.fail_stop = False
        self.starts = []
        self.paused = set()
        self.pause_calls = []
        self.resume_calls = []

    def pause_generation(self, replica, *, timeout=30):
        self.pause_calls.append(replica.replica_id)
        self.paused.add(replica.replica_id)

    def resume_generation(self, replica, *, timeout=30):
        self.resume_calls.append(replica.replica_id)
        self.paused.discard(replica.replica_id)

    def start(self, replica, spec):
        self.starts.append((replica.replica_id, spec.replica.model, replica.serving_policy_version))
        self.processes.add(replica.replica_id)
        if self.fail_launch:
            raise RuntimeError("launch failed")

    def alive(self, replica):
        return replica.replica_id in self.processes

    def health(self, replica):
        return self.healthy

    def warmup(self, replica, spec):
        return replica.replica_id not in self.paused

    def idle(self, replica):
        return replica.replica_id not in self.busy

    def stop(self, replica):
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.processes.discard(replica.replica_id)
        self.paused.discard(replica.replica_id)
        self.stopped.append(replica.replica_id)


@pytest.fixture(params=[8, 2], ids=["four-nodes-tp8", "one-node-tp2"])
def runtime(request):
    clock, driver = Clock(), FakeDriver()
    tp = request.param
    slots = [
        ResourceSlot(
            str(i),
            f"node{i if tp == 8 else 0}",
            f"host{i if tp == 8 else 0}",
            tp,
            port=8000 + i,
        )
        for i in range(4)
    ]
    controller = Controller(StaticResourceProvider(slots), driver, clock)
    spec = DeploymentSpec(
        "test", ReplicaSpec("model", tensor_parallel_size=tp), drain_grace_period=10, lease_ttl=20
    )
    controller.ensure_deployment(spec)
    return controller, controller.deployments["test"], driver, clock


def ready(runtime):
    controller, manager, _, _ = runtime
    controller.reconcile()
    controller.reconcile()
    return manager.traffic


def test_only_ready_receives_routes(runtime):
    controller, manager, _, _ = runtime
    controller.reconcile()
    with pytest.raises(NoReadyReplica):
        manager.traffic.acquire("a")
    controller.reconcile()
    assert manager.traffic.acquire("a").serving_policy_version == 0


def test_cold_warmup_timeout_retries_without_releasing_slot_or_routing(runtime):
    controller, manager, driver, _ = runtime
    controller.reconcile()
    first = set()

    def warmup(replica, spec):
        if replica.replica_id not in first:
            first.add(replica.replica_id)
            raise TimeoutError("cold first-token compilation")
        return True

    driver.warmup = warmup
    allocated = manager.resources.allocated
    controller.reconcile()
    assert manager.resources.allocated == allocated and not driver.stopped
    assert all(r.state == S.WARMING for r in manager.traffic.replicas.values())
    with pytest.raises(NoReadyReplica):
        manager.traffic.acquire("during-compile")
    controller.reconcile()
    assert all(r.state == S.READY and r.error is None for r in manager.traffic.replicas.values())


def test_warmup_retries_remain_bounded_by_startup_deadline(runtime):
    controller, manager, driver, clock = runtime
    controller.reconcile()
    originals = list(manager.traffic.replicas.values())

    def timeout(*_):
        raise TimeoutError("still compiling")

    driver.warmup = timeout
    controller.reconcile()
    clock.now += manager.spec.replica.startup_timeout + 1
    controller.reconcile()
    assert all(r.state == S.FAILED and r.resource is None for r in originals)
    assert all(r.replica_id in driver.stopped for r in originals)


def test_invalid_warmup_response_is_not_retried(runtime):
    controller, manager, driver, _ = runtime
    controller.reconcile()
    originals = list(manager.traffic.replicas.values())

    def invalid(*_):
        raise ValueError("missing token logprobs")

    driver.warmup = invalid
    controller.reconcile()
    assert all(r.state == S.FAILED and r.error == "missing token logprobs" for r in originals)


@pytest.mark.parametrize("state", [S.STARTING, S.SYNCING, S.WARMING])
def test_shutdown_stops_startup_without_waiting_for_readiness(runtime, state):
    controller, manager, driver, _ = runtime
    controller.reconcile()
    for replica in manager.traffic.replicas.values():
        replica.transition(state)

    def forbidden(*args):
        pytest.fail("shutdown must not probe or warm a startup replica")

    driver.health = driver.warmup = forbidden
    controller.close(timeout=0.1)
    assert not driver.processes and manager.resources.allocated == 0
    assert all(r.state == S.TERMINATED for r in manager.traffic.replicas.values())


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
