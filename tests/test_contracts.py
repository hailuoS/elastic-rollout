"""Resource and routing edge cases independent of serving processes."""

from dataclasses import asdict

import pytest

from elastic_rollout.api import (
    DeploymentSpec,
    ReplicaDescriptor,
    ReplicaSpec,
    ReplicaState,
    ResourceSlot,
)
from elastic_rollout.resource import NoCapacity, StaticResourceProvider
from elastic_rollout.traffic import NoReadyReplica, TrafficManager


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_invalid_replica_count(value):
    with pytest.raises(ValueError):
        DeploymentSpec("test", ReplicaSpec("model"), desired_replicas=value)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_lease_ttl(value):
    with pytest.raises(ValueError):
        DeploymentSpec("test", ReplicaSpec("model"), lease_ttl=value)


@pytest.mark.parametrize("name", ["", "a/b", "a b", "a?b"])
def test_deployment_identifier_rejected(name):
    with pytest.raises(ValueError):
        DeploymentSpec(name, ReplicaSpec("model"))


@pytest.mark.parametrize("port", [0, 65536, True, 80.5])
def test_invalid_serving_port(port):
    with pytest.raises(ValueError):
        ResourceSlot("slot", "node", "host", 2, port=port)


def test_deployment_serialization_roundtrip():
    spec = DeploymentSpec(
        "test",
        ReplicaSpec("model", vllm_args=("--dtype", "bfloat16")),
        desired_replicas=0,
        drain_grace_period=0,
    )
    assert DeploymentSpec.from_dict(asdict(spec)) == spec


def test_ipv6_endpoint_uses_brackets():
    assert ResourceSlot("s", "n", "::1", 2).endpoint == "http://[::1]:8000"


def test_duplicate_slot_ids_rejected():
    with pytest.raises(ValueError, match="duplicate slot_id"):
        StaticResourceProvider(
            [
                ResourceSlot("s", "n1", "h1", 2),
                ResourceSlot("s", "n2", "h2", 2),
            ]
        )


def test_resource_exhaustion_and_stale_release():
    provider = StaticResourceProvider([ResourceSlot("s", "n", "h", 2)])
    first = provider.acquire("first")
    assert provider.acquire("first") == first
    with pytest.raises(NoCapacity):
        provider.acquire("second")
    assert provider.release(first)
    second = provider.acquire("second")
    assert second.lease_id != first.lease_id
    assert not provider.release(first)
    assert provider.allocated == 1
    assert provider.release(second)
    assert provider.allocated == 0


@pytest.fixture
def routing():
    now = [100.0]
    traffic = TrafficManager(
        DeploymentSpec("test", ReplicaSpec("model"), lease_ttl=10, sticky_cache_size=2),
        clock=lambda: now[0],
    )
    for name in ("a", "b"):
        traffic.replicas[name] = ReplicaDescriptor(
            name,
            "test",
            3,
            state=ReplicaState.READY,
            endpoint=f"http://{name}:8000",
        )
    return traffic, now


@pytest.mark.parametrize("key", ["", None, 42])
def test_invalid_routing_key(routing, key):
    with pytest.raises(ValueError):
        routing[0].acquire(key)


def test_sticky_lru_is_bounded_without_invalidating_leases(routing):
    traffic, _ = routing
    first = traffic.acquire("first")
    second = traffic.acquire("second")
    traffic.acquire("first")
    third = traffic.acquire("third")
    assert list(traffic.sticky) == ["first", "third"]
    assert second.lease_id in traffic.leases
    assert first.serving_policy_version == third.serving_policy_version == 3
    assert sum(traffic.inflight.values()) == 4


def test_deadline_routes_sticky_key_away_from_draining_replica(routing):
    traffic, now = routing
    first = traffic.acquire("conversation")
    replica = traffic.replicas[first.replica_id]
    replica.state = ReplicaState.DRAINING
    replica.drain_deadline = now[0] + 5
    assert traffic.acquire("conversation").replica_id == first.replica_id
    now[0] += 5
    assert traffic.acquire("conversation").replica_id != first.replica_id
    assert traffic.renew(first.lease_id).replica_id == first.replica_id


def test_ttl_boundary_does_not_allow_resurrection(routing):
    traffic, now = routing
    lease = traffic.acquire("key")
    now[0] = lease.expires_at
    with pytest.raises(KeyError):
        traffic.renew(lease.lease_id)
    assert not traffic.release(lease.lease_id)
    assert sum(traffic.inflight.values()) == 0


def test_no_ready_replica_preserves_existing_lease(routing):
    traffic, _ = routing
    lease = traffic.acquire("existing")
    for replica in traffic.replicas.values():
        replica.state = ReplicaState.DRAINING
        replica.drain_deadline = 110.0
    with pytest.raises(NoReadyReplica):
        traffic.acquire("new")
    assert traffic.renew(lease.lease_id).lease_id == lease.lease_id
    assert traffic.release(lease.lease_id)


def test_invalidation_affects_only_failed_replica(routing):
    traffic, _ = routing
    victim = traffic.acquire("victim")
    survivor = traffic.acquire("survivor")
    traffic.replicas[victim.replica_id].state = ReplicaState.FAILED
    assert traffic.invalidate_replica(victim.replica_id) == 1
    assert traffic.invalidate_replica(victim.replica_id) == 0
    assert set(traffic.leases) == {survivor.lease_id}
    assert "victim" not in traffic.sticky
    assert not traffic.release(victim.lease_id)
    assert traffic.inflight[victim.replica_id] == 0
    assert traffic.renew(survivor.lease_id).replica_id == survivor.replica_id
