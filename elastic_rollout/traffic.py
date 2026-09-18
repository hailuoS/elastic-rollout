"""Single authoritative sticky routing table and expiring request leases."""

import time
from collections import OrderedDict
from dataclasses import replace
from threading import RLock
from typing import Callable
from uuid import uuid4

from .api import DeploymentSpec, ReplicaDescriptor, ReplicaState, RouteLease


class NoReadyReplica(RuntimeError):
    pass


class TrafficManager:
    def __init__(self, spec: DeploymentSpec, clock: Callable[[], float] = time.time):
        self.spec = spec
        self.clock = clock
        self.lock = RLock()
        self.replicas: dict[str, ReplicaDescriptor] = {}
        self.sticky: OrderedDict[str, str] = OrderedDict()
        self.leases: dict[str, RouteLease] = {}
        self.inflight: dict[str, int] = {}
        self.accepting_routes = True

    def pause_routes(self) -> None:
        with self.lock:
            self.accepting_routes = False

    def resume_routes(self) -> None:
        with self.lock:
            self.accepting_routes = True

    def expire(self) -> None:
        with self.lock:
            for lease in list(self.leases.values()):
                if lease.expires_at <= self.clock():
                    self.release(lease.lease_id)

    def acquire(self, routing_key: str) -> RouteLease:
        if not isinstance(routing_key, str) or not routing_key:
            raise ValueError("routing_key must be a nonempty opaque string")
        with self.lock:
            self.expire()
            if not self.accepting_routes:
                raise NoReadyReplica("policy update in progress or failed; routing is paused")
            replica = self.replicas.get(self.sticky.get(routing_key, ""))
            if replica is not None and not (
                replica.state == ReplicaState.READY
                or (
                    replica.state == ReplicaState.DRAINING and self.clock() < replica.drain_deadline
                )
            ):
                replica = None
            if replica is None:
                ready = [r for r in self.replicas.values() if r.state == ReplicaState.READY]
                if not ready:
                    raise NoReadyReplica("no READY replica; retry after capacity becomes ready")
                replica = min(ready, key=lambda r: self.inflight.get(r.replica_id, 0))
            self.sticky[routing_key] = replica.replica_id
            self.sticky.move_to_end(routing_key)
            while len(self.sticky) > self.spec.sticky_cache_size:
                self.sticky.popitem(last=False)
            lease = RouteLease(
                uuid4().hex,
                self.spec.deployment_id,
                routing_key,
                replica.replica_id,
                replica.endpoint,
                replica.serving_policy_version,
                self.clock() + self.spec.lease_ttl,
            )
            self.leases[lease.lease_id] = lease
            self.inflight[replica.replica_id] = self.inflight.get(replica.replica_id, 0) + 1
            return lease

    def release(self, lease_id: str) -> bool:
        with self.lock:
            lease = self.leases.pop(lease_id, None)
            if lease is None:
                return False
            self.inflight[lease.replica_id] -= 1
            return True

    def renew(self, lease_id: str) -> RouteLease:
        with self.lock:
            self.expire()
            lease = self.leases[lease_id]
            replica = self.replicas[lease.replica_id]
            if replica.state not in (ReplicaState.READY, ReplicaState.DRAINING):
                raise NoReadyReplica("replica is no longer serving")
            lease = replace(lease, expires_at=self.clock() + self.spec.lease_ttl)
            self.leases[lease_id] = lease
            return lease

    def forget_replica(self, replica_id: str) -> None:
        with self.lock:
            for key, target in list(self.sticky.items()):
                if target == replica_id:
                    del self.sticky[key]

    def invalidate_replica(self, replica_id: str) -> int:
        """Remove leases only after confirmed failure, never during normal drain."""
        with self.lock:
            if self.replicas[replica_id].state != ReplicaState.FAILED:
                raise ValueError("only FAILED replicas can invalidate active leases")
            self.forget_replica(replica_id)
            leases = [
                lease.lease_id for lease in self.leases.values() if lease.replica_id == replica_id
            ]
            for lease_id in leases:
                self.release(lease_id)
            return len(leases)
