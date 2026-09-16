"""Desired-state reconciliation. Slow engine operations never hold the routing lock."""

import logging
import threading
import time
from dataclasses import asdict, replace
from typing import Callable, Protocol
from uuid import uuid4

from .api import DeploymentSpec, ReplicaDescriptor, ReplicaState, nonnegative_int
from .resource import NoCapacity, StaticResourceProvider
from .traffic import TrafficManager

logger = logging.getLogger(__name__)
S = ReplicaState
STARTUP = (S.ALLOCATED, S.STARTING, S.SYNCING, S.WARMING)
DRAIN = (S.DRAINING, S.QUIESCED, S.STOPPING)
TERMINAL = (S.FAILED, S.TERMINATED)


class ServiceDriver(Protocol):
    def start(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> None: ...
    def alive(self, replica: ReplicaDescriptor) -> bool: ...
    def health(self, replica: ReplicaDescriptor) -> bool: ...
    def warmup(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> bool: ...
    def idle(self, replica: ReplicaDescriptor) -> bool: ...
    def stop(self, replica: ReplicaDescriptor) -> None: ...


class ReplicaManager:
    def __init__(
        self,
        spec: DeploymentSpec,
        resources: StaticResourceProvider,
        driver: ServiceDriver,
        clock: Callable[[], float] = time.time,
    ):
        self.spec = spec
        self.desired = spec.desired_replicas
        self.resources = resources
        self.driver = driver
        self.clock = clock
        self.traffic = TrafficManager(spec, clock)
        self.mutation_lock = threading.Lock()
        self.last_error: str | None = None

    def _transition(self, replica: ReplicaDescriptor, state: ReplicaState) -> None:
        with self.traffic.lock:
            replica.transition(state)

    def _cleanup(self, replica: ReplicaDescriptor) -> None:
        # A failed stop quarantines its slot. Never reuse possibly occupied devices.
        try:
            self.driver.stop(replica)
        except Exception as exc:
            replica.error = f"cleanup pending: {exc}"
            self.last_error = replica.error
            return
        if replica.resource is not None:
            self.resources.release(replica.resource)
            replica.resource = None
        if replica.state != S.FAILED:
            self._transition(replica, S.TERMINATED)

    def _fail(self, replica: ReplicaDescriptor, error: Exception) -> None:
        replica.error = str(error)
        self.last_error = str(error)
        self._transition(replica, S.FAILED)
        self.traffic.forget_replica(replica.replica_id)
        self._cleanup(replica)

    def _create(self) -> bool:
        replica = ReplicaDescriptor(
            uuid4().hex, self.spec.deployment_id, self.spec.replica.policy_version
        )
        try:
            replica.resource = self.resources.acquire(replica.replica_id)
        except NoCapacity as exc:
            self.last_error = str(exc)
            return False
        replica.endpoint = replica.resource.slot.endpoint
        with self.traffic.lock:
            self.traffic.replicas[replica.replica_id] = replica
        try:
            self._transition(replica, S.ALLOCATED)
            replica.started_at = self.clock()
            self._transition(replica, S.STARTING)
            self.driver.start(replica, self.spec)
        except Exception as exc:
            self._fail(replica, exc)
        return True

    def reconcile(self) -> None:
        # A deployment has only one lifecycle mutation in progress. API handlers
        # merely change desired state; they never create/destroy processes.
        with self.mutation_lock:
            self.traffic.expire()
            for replica in list(self.traffic.replicas.values()):
                try:
                    if replica.state == S.FAILED:
                        if replica.resource is not None:
                            self._cleanup(replica)
                        continue
                    if replica.state == S.TERMINATED:
                        continue
                    if replica.state == S.STOPPING:
                        self._cleanup(replica)
                        continue
                    if not self.driver.alive(replica):
                        raise RuntimeError("replica process exited")
                    if replica.state in STARTUP:
                        if self.clock() - replica.started_at >= self.spec.replica.startup_timeout:
                            raise TimeoutError("replica startup/health timeout")
                        if not self.driver.health(replica):
                            continue
                        if replica.state == S.STARTING:
                            # Fixed policy loaded by vLLM at startup; no weight transport.
                            self._transition(replica, S.SYNCING)
                            self._transition(replica, S.WARMING)
                        if self.driver.warmup(replica, self.spec):
                            self._transition(replica, S.READY)
                    elif replica.state in (S.READY, S.DRAINING):
                        replica.health_failures = (
                            0 if self.driver.health(replica) else replica.health_failures + 1
                        )
                        if replica.health_failures >= 3:
                            raise RuntimeError("replica failed three consecutive health probes")
                    if replica.state == S.DRAINING and self.clock() >= replica.drain_deadline:
                        self.traffic.forget_replica(replica.replica_id)
                        with self.traffic.lock:
                            no_leases = self.traffic.inflight.get(replica.replica_id, 0) == 0
                        # TTL expiry alone is not proof that server generation finished.
                        if no_leases and self.driver.idle(replica):
                            self._transition(replica, S.QUIESCED)
                            self._transition(replica, S.STOPPING)
                            self._cleanup(replica)
                except Exception as exc:
                    self._fail(replica, exc)

            with self.traffic.lock:
                live = [r for r in self.traffic.replicas.values() if r.state not in TERMINAL]
                # Finish one scale direction before applying the newest desired count.
                if any(r.state in STARTUP + DRAIN for r in live):
                    return
                desired = self.desired
                ready = [r for r in live if r.state == S.READY]
                excess = len(ready) - desired
                if excess > 0:
                    victims = sorted(
                        ready, key=lambda r: self.traffic.inflight.get(r.replica_id, 0)
                    )
                    for replica in victims[:excess]:
                        replica.drain_deadline = self.clock() + self.spec.drain_grace_period
                        replica.transition(S.DRAINING)
                    return
            for _ in range(max(0, desired - len(ready))):
                if not self._create():
                    break

    def status(self) -> dict:
        with self.traffic.lock:
            self.traffic.expire()
            replicas = list(self.traffic.replicas.values())
            return {
                "spec": asdict(self.spec),
                "desired_replicas": self.desired,
                "ready_replicas": sum(r.state == S.READY for r in replicas),
                "available_replicas": sum(r.state in (S.READY, S.DRAINING) for r in replicas),
                "total_replicas": sum(r.state not in TERMINAL for r in replicas),
                "inflight": sum(self.traffic.inflight.values()),
                "last_error": self.last_error,
            }


class Controller:
    def __init__(
        self,
        resources: StaticResourceProvider,
        driver: ServiceDriver,
        clock: Callable[[], float] = time.time,
    ):
        self.resources, self.driver, self.clock = resources, driver, clock
        self.deployments: dict[str, ReplicaManager] = {}
        self._lock = threading.RLock()
        self._closed = False

    def ensure_deployment(self, spec: DeploymentSpec) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("controller is closing")
            if spec.deployment_id in self.deployments:
                old = self.deployments[spec.deployment_id].spec
                if replace(old, desired_replicas=spec.desired_replicas) != spec:
                    raise ValueError("deployment configuration is immutable in MVP-0")
            else:
                self.deployments[spec.deployment_id] = ReplicaManager(
                    spec, self.resources, self.driver, self.clock
                )
            # Ensure is idempotent; it must not undo a separate SetDesiredReplicas.
            return self.deployments[spec.deployment_id].status()

    def set_desired_replicas(self, deployment_id: str, count: int) -> dict:
        nonnegative_int(count, "desired_replicas")
        with self._lock:
            if self._closed and count:
                raise RuntimeError("controller is closing")
            manager = self.deployments[deployment_id]
            with manager.traffic.lock:
                manager.desired = count
            return manager.status()

    def reconcile(self) -> None:
        with self._lock:
            managers = list(self.deployments.values())
        for manager in managers:
            manager.reconcile()

    def close(self, timeout: float = 600) -> None:
        with self._lock:
            self._closed = True
            for manager in self.deployments.values():
                self.set_desired_replicas(manager.spec.deployment_id, 0)
        deadline = time.monotonic() + timeout
        while True:
            self.reconcile()
            if self.resources.allocated == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("cleanup incomplete; resources remain quarantined or draining")
            time.sleep(0.1)


class Reconciler:
    def __init__(self, controller: Controller, interval: float = 1.0):
        self.controller, self.interval = controller, interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("reconciler already started")
        self._thread = threading.Thread(target=self._run, daemon=True, name="elastic-reconciler")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.controller.reconcile()
            except Exception:
                logger.exception("reconcile failed; retrying next tick")
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
