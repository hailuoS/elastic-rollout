"""Desired-state reconciliation. Slow engine operations never hold the routing lock."""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Callable, Protocol
from uuid import uuid4

from .api import (
    DeploymentSpec,
    PolicySnapshot,
    PolicyUpdateDescriptor,
    PolicyUpdateState,
    ReplicaDescriptor,
    ReplicaState,
    nonnegative_int,
)
from .api import (
    PolicyPreparationKind as K,
)
from .resource import NoCapacity, StaticResourceProvider
from .traffic import TrafficManager
from .weight_transfer import RECOVERY_REQUIRED, WeightTransferControl

logger = logging.getLogger(__name__)
S = ReplicaState
STARTUP = (S.ALLOCATED, S.STARTING, S.SYNCING, S.WARMING)
DRAIN = (S.DRAINING, S.QUIESCED, S.STOPPING)
TERMINAL = (S.FAILED, S.TERMINATED)
U = PolicyUpdateState


class PolicyConflict(RuntimeError):
    pass


class GenerationControlTimeout(TimeoutError):
    pass


class ServiceDriver(Protocol):
    @property
    def weight_transfer_control(self) -> WeightTransferControl: ...

    def start(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> None: ...
    def alive(self, replica: ReplicaDescriptor) -> bool: ...
    def health(self, replica: ReplicaDescriptor) -> bool: ...
    def warmup(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> bool: ...
    def idle(self, replica: ReplicaDescriptor) -> bool: ...
    def stop(self, replica: ReplicaDescriptor) -> None: ...
    def pause_generation(self, replica: ReplicaDescriptor, *, timeout: float = 30) -> None: ...
    def resume_generation(self, replica: ReplicaDescriptor, *, timeout: float = 30) -> None: ...


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
        # Serializes physical generation control with process incarnation changes.
        # Lock order: generation -> mutation -> traffic. Serving lease operations
        # only take traffic, never either lifecycle/control lock.
        self.generation_lock = threading.Lock()
        self.generation_paused = False
        self._generation_control_failed = False
        self._generation_pause_verified = False
        self._generation_timeout_epoch = 0
        self.last_error: str | None = None
        self.committed_policy = PolicySnapshot(spec.replica.policy_version, spec.replica.model)
        self.bootstrap_snapshot = self.committed_policy
        self.serving_policy_version = spec.replica.policy_version
        from .weight_transfer import TransferCoordinator

        self.transfer = TransferCoordinator(self)
        self.policy_update: PolicyUpdateDescriptor | None = None
        self.policy_updates: dict[str, PolicyUpdateDescriptor] = {}
        self.closing = False

    @contextmanager
    def _control_mutation(self):
        # HTTP handlers never wait for a process stop, start or health probe.
        if not self.mutation_lock.acquire(blocking=False):
            raise PolicyConflict("deployment lifecycle mutation in progress")
        try:
            with self.traffic.lock:
                yield
        finally:
            self.mutation_lock.release()

    def _effective_spec(self, model_path: str) -> DeploymentSpec:
        return replace(self.spec, replica=replace(self.spec.replica, model=model_path))

    def _update_blocks_topology(self) -> bool:
        return self.policy_update is not None and self.policy_update.state != U.COMMITTED

    def _stable_for_policy_update(self, replicas: list) -> bool:
        """The topology predicate shared by begin and the serving readiness view.

        Call with traffic.lock held. Begin additionally serializes mutations and
        checks the requested update kind and target version.
        """
        return (
            not self._update_blocks_topology()
            and self.desired > 0
            and not any(r.state in STARTUP + DRAIN for r in replicas)
            and not any(r.state in TERMINAL and r.resource is not None for r in replicas)
            and sum(r.state == S.READY for r in replicas) == self.desired
        )

    def set_desired(self, count: int) -> dict:
        # Preserve MVP-0 desired-state changes during normal reconciliation.
        # Begin/commit also hold this lock, so the policy gate is atomic here.
        with self.traffic.lock:
            if self.spec.replica.weight_sync == "hccl" and count != self.desired:
                raise PolicyConflict("Direct HCCL requires fixed topology; scaling is unsupported")
            if "generation" in self.traffic.route_pause_reasons:
                raise PolicyConflict("generation paused; topology is locked")
            if self._update_blocks_topology():
                raise PolicyConflict("policy update active or failed; topology is locked")
            self.desired = count
            return self.status()

    def pause_generation(self) -> dict:
        return self._set_generation_paused(True)

    def resume_generation(self) -> dict:
        return self._set_generation_paused(False)

    def _generation_timeout(self, action: str) -> GenerationControlTimeout:
        with self.traffic.lock:
            self.traffic.pause_routes("generation")
            self._generation_control_failed = True
            self._generation_timeout_epoch += 1
            inflight = {rid: n for rid, n in self.traffic.inflight.items() if n}
            self.last_error = (
                f"generation {action} timed out after {self.spec.generation_control_timeout_s}s; "
                f"inflight={sum(inflight.values())}; replica_ids={sorted(inflight)}"
            )
            return GenerationControlTimeout(self.last_error)

    def _generation_remaining(self, deadline: float, action: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._generation_timeout(action)
        return remaining

    @contextmanager
    def _generation_operation(self, deadline: float, action: str):
        if not self.generation_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise self._generation_timeout(action)
        try:
            yield
        finally:
            self.generation_lock.release()

    def _set_generation_paused(self, paused: bool) -> dict:
        action = "pause" if paused else "resume"
        deadline = time.monotonic() + self.spec.generation_control_timeout_s
        with self._generation_operation(deadline, action):
            self._generation_remaining(deadline, action)
            with self.traffic.lock:
                if self.closing:
                    raise PolicyConflict("controller is closing")
                if not paused and self._external_update() and self._update_blocks_topology():
                    raise PolicyConflict("external update must COMMIT before generation resume")
                gated = "generation" in self.traffic.route_pause_reasons
                if (
                    paused
                    and self._generation_pause_verified
                    and not self._generation_control_failed
                ) or (not paused and not gated):
                    return self.status()
                self.traffic.pause_routes("generation")
                timeout_epoch = self._generation_timeout_epoch
                targets = [
                    r for r in self.traffic.replicas.values() if r.state in (S.READY, S.DRAINING)
                ]
            logger.info(
                "generation %s requested deployment_id=%s targets=%s reason=generation",
                action,
                self.spec.deployment_id,
                len(targets),
            )
            try:
                control = self.driver.pause_generation if paused else self.driver.resume_generation
                while True:
                    for replica in targets:
                        control(
                            replica, timeout=min(30, self._generation_remaining(deadline, action))
                        )
                    self._generation_remaining(deadline, action)
                    with self.traffic.lock:
                        inflight = sum(self.traffic.inflight.values())
                    if not paused or inflight == 0:
                        break
                    # Re-abort requests arriving after the previous pause. Keep
                    # leases valid so their clients can receive partials and release.
                    time.sleep(min(0.1, self._generation_remaining(deadline, action)))
            except Exception as exc:
                with self.traffic.lock:
                    self._generation_control_failed = True
                    self.last_error = f"generation {action} failed: {exc}"
                # The request may have reached vLLM even if its response was lost.
                # Keep the gate; explicit pause/resume retries address current targets.
                if isinstance(exc, GenerationControlTimeout):
                    raise
                raise RuntimeError(self.last_error) from exc
            with self.traffic.lock:
                # A waiting caller may have timed out and closed admission while
                # this operation was in I/O. Never undo that fail-closed decision.
                if timeout_epoch != self._generation_timeout_epoch:
                    raise GenerationControlTimeout(self.last_error)
                self.generation_paused = paused
                self._generation_pause_verified = paused
                self._generation_control_failed = False
                if not paused:
                    self.traffic.resume_routes("generation")
                status = self.status()
            logger.info(
                "generation %s complete deployment_id=%s targets=%s route_pause_reasons=%s",
                action,
                self.spec.deployment_id,
                len(targets),
                status["route_pause_reasons"],
            )
            return status

    def _external_update(self) -> bool:
        return bool(self.policy_update and self.policy_update.preparation_kind == K.EXTERNAL_HCCL)

    def begin_policy_update(
        self,
        snapshot_path: str | None = None,
        preparation_kind: str = K.SNAPSHOT_RESTART,
        target_version: int | None = None,
    ) -> dict:
        kind = K(preparation_kind)
        if kind == K.EXTERNAL_HCCL:
            nonnegative_int(target_version, "target_version")
            if self.spec.replica.weight_sync != "hccl":
                raise PolicyConflict("deployment is not HCCL capable")
            if snapshot_path is not None:
                raise ValueError("EXTERNAL_HCCL has no snapshot_path")
        elif not isinstance(snapshot_path, str) or not snapshot_path.strip():
            raise ValueError("snapshot_path must be a non-empty string")
        with self._control_mutation():
            if (
                kind == K.EXTERNAL_HCCL
                and self.transfer.transport
                and self.transfer.transport.state == "POISONED"
            ):
                raise PolicyConflict(RECOVERY_REQUIRED)
            if kind == K.EXTERNAL_HCCL and target_version <= self.serving_policy_version:
                raise ValueError("target_version must exceed serving_policy_version")
            replicas = list(self.traffic.replicas.values())
            if self._update_blocks_topology():
                raise PolicyConflict("policy update active or failed")
            if self.desired <= 0:
                raise PolicyConflict("policy update requires at least one READY replica")
            if not self._stable_for_policy_update(replicas):
                raise PolicyConflict("deployment must be stable before policy update")
            update = PolicyUpdateDescriptor(
                uuid4().hex,
                self.spec.deployment_id,
                self.serving_policy_version,
                snapshot_path,
                started_at=self.clock(),
                target_replica_ids=tuple(r.replica_id for r in replicas if r.state == S.READY),
                deadline_at=self.clock() + self.spec.policy_update_runtime_timeout_s,
                preparation_kind=kind,
                target_version=target_version,
            )
            self.policy_update = update
            if kind == K.EXTERNAL_HCCL:
                self.transfer.session = None
            self.policy_updates[update.update_id] = update
            self.traffic.pause_routes("policy_update")
            logger.info("policy update %s requested; routing paused", update.update_id)
            return asdict(update)

    def get_policy_update(self, update_id: str) -> dict:
        with self.traffic.lock:
            return asdict(self.policy_updates[update_id])

    def commit_policy_update(self, update_id: str, policy_version: int) -> dict:
        nonnegative_int(policy_version, "policy_version")
        with self._control_mutation():
            update = self.policy_updates[update_id]
            if update.state == U.COMMITTED and update.committed_version == policy_version:
                return asdict(update)
            if update is self.policy_update and self._check_update_deadline():
                raise PolicyConflict(f"policy update FAILED: {update.error}")
            if update is not self.policy_update or update.state != U.PREPARED:
                raise PolicyConflict("policy update must be PREPARED before commit")
            if policy_version <= self.serving_policy_version:
                raise ValueError("policy_version must exceed the current committed version")
            live = [r for r in self.traffic.replicas.values() if r.state not in TERMINAL]
            if self._external_update():
                if policy_version != update.target_version:
                    raise ValueError("commit version must equal bound target_version")
                if set(r.replica_id for r in live) != set(update.target_replica_ids) or any(
                    r.state != S.READY or r.loaded_policy_version != policy_version for r in live
                ):
                    raise PolicyConflict("frozen targets must all have loaded target_version")
            elif not self._candidate_ready(live):
                raise PolicyConflict("candidate replicas are no longer all READY")
            if not self._external_update():
                self.bootstrap_snapshot = PolicySnapshot(policy_version, update.snapshot_path)
            self.serving_policy_version = policy_version
            # Compatibility view: path is always the durable bootstrap, never a
            # claim that an in-memory HCCL version has been written to disk.
            self.committed_policy = PolicySnapshot(
                policy_version, self.bootstrap_snapshot.model_path
            )
            for replica in live:
                replica.serving_policy_version = policy_version
                replica.loaded_policy_version = policy_version
            self.traffic.sticky.clear()
            update.committed_version = policy_version
            self._update_transition(U.COMMITTED)
            self.traffic.resume_routes("policy_update")
            logger.info(
                "policy version %s COMMITTED; route_pause_reasons=%s",
                policy_version,
                self.traffic.route_pause_reasons,
            )
            return asdict(update)

    def _update_transition(self, state: PolicyUpdateState) -> None:
        with self.traffic.lock:
            self.policy_update.state = state
            self.policy_update.history.append(state.value)
        logger.info("policy update %s %s", self.policy_update.update_id, state.value)

    def _fail_update(self, error: Exception) -> None:
        with self.traffic.lock:
            if self.policy_update.state in (U.FAILED, U.COMMITTED):
                return
            self.policy_update.error = str(error)
            self.last_error = str(error)
            self.traffic.pause_routes("policy_update")
            if self._external_update():
                self.traffic.pause_routes("generation")
                self.transfer.fail(str(error))
            self._update_transition(U.FAILED)
        logger.error("policy update FAILED: %s", error)

    def _check_update_deadline(self) -> bool:
        update = self.policy_update
        if update is None or update.state == U.COMMITTED or self.closing:
            return False
        if update.state != U.FAILED and self.clock() >= update.deadline_at:
            self._fail_update(TimeoutError("policy update runtime deadline exceeded"))
        return update.state == U.FAILED

    def _candidate_ready(self, live: list[ReplicaDescriptor]) -> bool:
        return (
            bool(live)
            and len(live) == self.desired
            and all(
                r.state == S.READY and r.loaded_model_path == self.policy_update.snapshot_path
                for r in live
            )
        )

    def _reconcile_policy(self) -> None:
        update = self.policy_update
        if self._external_update():
            return  # Explicit transfer operations own preparation; never restart targets.
        live = [r for r in self.traffic.replicas.values() if r.state not in TERMINAL]
        if any(
            r.state == S.FAILED and r.resource is not None for r in self.traffic.replicas.values()
        ):
            raise RuntimeError("policy update cannot proceed: replica resource quarantined")
        if update.state == U.QUIESCING:
            with self.traffic.lock:
                inflight = sum(self.traffic.inflight.values())
            if inflight:
                logger.info("policy update waiting for %s inflight requests", inflight)
                return
            if not all(self.driver.idle(r) for r in live):
                return
            self._update_transition(U.RESTARTING)
            self._prepare_policy_via_snapshot_restart(live)
            return
        if update.state in (U.RESTARTING, U.VERIFYING, U.PREPARED):
            if self._candidate_ready(live):
                if update.state == U.RESTARTING:
                    self._update_transition(U.VERIFYING)
                elif update.state == U.VERIFYING:
                    self._update_transition(U.PREPARED)
                return
            if update.state != U.RESTARTING:
                self._update_transition(U.RESTARTING)
            for _ in range(max(0, self.desired - len(live))):
                if self._check_update_deadline():
                    return
                if not self._create():
                    break

    def _prepare_policy_via_snapshot_restart(self, targets: list[ReplicaDescriptor]) -> None:
        """Physical preparation only; the transaction retains the gate and commit.

        Fresh processes are running (not physically paused). The generation route
        gate survives restart, so warmup can run while logical continuations wait.
        Snapshot health/warmup and candidate replacement reuse normal reconciliation.
        """
        snapshot_path = self.policy_update.snapshot_path
        for replica in targets:
            if self._check_update_deadline():
                return
            logger.info("restarting replica %s from snapshot %s", replica.replica_id, snapshot_path)
            # Preserve the resource lease/endpoint, never a request's RouteLease.
            # A stop error is fatal: never start on possibly occupied devices.
            self.driver.stop(replica)
            replica.loaded_model_path = snapshot_path
            replica.loaded_policy_version = None  # Snapshot version is supplied at commit.
            replica.started_at = self.clock()
            replica.health_failures = 0
            self._transition(replica, S.STARTING)
            try:
                self.driver.start(replica, self._effective_spec(snapshot_path))
            except Exception as exc:
                self._fail(replica, exc)

    def _transition(self, replica: ReplicaDescriptor, state: ReplicaState) -> None:
        with self.traffic.lock:
            if state in (S.STARTING, S.READY):
                self._generation_pause_verified = False
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
        update = self.policy_update
        if (
            self._external_update()
            and self._update_blocks_topology()
            and replica.replica_id in update.target_replica_ids
        ):
            self._fail_update(error)
        candidate_failure = (
            not self.closing
            and update is not None
            and update.state in (U.RESTARTING, U.VERIFYING, U.PREPARED)
            and replica.loaded_model_path == update.snapshot_path
            and replica.state not in TERMINAL
        )
        replica.error = str(error)
        self.last_error = str(error)
        # Renew/acquire must not see FAILED with still-active leases. Cleanup may
        # block or fail, so invalidate before stopping/quarantining the process.
        with self.traffic.lock:
            self._transition(replica, S.FAILED)
            invalidated = self.traffic.invalidate_replica(replica.replica_id)
        logger.info(
            "replica failed replica_id=%s invalidated_leases=%s",
            replica.replica_id,
            invalidated,
        )
        self._cleanup(replica)
        if candidate_failure:
            with self.traffic.lock:
                update.candidate_failure_count += 1
                if update.candidate_failure_count >= self.spec.candidate_start_max_failures:
                    self._fail_update(
                        RuntimeError(
                            f"candidate failure budget exhausted "
                            f"({update.candidate_failure_count}/"
                            f"{self.spec.candidate_start_max_failures}): {error}"
                        )
                    )

    def _create(self) -> bool:
        model_path = self.committed_policy.model_path
        candidate = self.policy_update is not None and self.policy_update.state in (
            U.RESTARTING,
            U.VERIFYING,
            U.PREPARED,
        )
        if candidate:
            model_path = self.policy_update.snapshot_path
        elif self.serving_policy_version != self.bootstrap_snapshot.version:
            self.last_error = "HCCL policy diverged from bootstrap; stale replacement disabled"
            return False
        replica = ReplicaDescriptor(
            uuid4().hex,
            self.spec.deployment_id,
            self.committed_policy.version,
            loaded_model_path=model_path,
            loaded_policy_version=self.bootstrap_snapshot.version,
        )
        try:
            replica.resource = self.resources.acquire(replica.replica_id)
        except NoCapacity as exc:
            self.last_error = str(exc)
            return False
        replica.endpoint = replica.resource.slot.endpoint
        replica.allocation = replica.resource.allocation
        with self.traffic.lock:
            self.traffic.replicas[replica.replica_id] = replica
        try:
            self._transition(replica, S.ALLOCATED)
            replica.started_at = self.clock()
            self._transition(replica, S.STARTING)
            logger.info(
                "new/replacement replica %s launching %s policy %s from %s",
                replica.replica_id,
                "candidate" if candidate else "committed",
                self.committed_policy.version,
                model_path,
            )
            self.driver.start(replica, self._effective_spec(model_path))
        except Exception as exc:
            self._fail(replica, exc)
        return True

    def reconcile(self) -> None:
        # A deployment has only one lifecycle mutation in progress. API handlers
        # merely change desired state; they never create/destroy processes.
        with self.generation_lock, self.mutation_lock:
            self.traffic.expire()
            if self._update_blocks_topology() and not self.closing:
                if self._check_update_deadline():
                    return
                if self.policy_update.state == U.REQUESTED and not self._external_update():
                    self._update_transition(U.QUIESCING)
            for replica in list(self.traffic.replicas.values()):
                if self._check_update_deadline():
                    return
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
                            # Model artifact is loaded by vLLM at process startup.
                            self._transition(replica, S.SYNCING)
                            self._transition(replica, S.WARMING)
                        if self.driver.warmup(
                            replica, self._effective_spec(replica.loaded_model_path)
                        ):
                            self._transition(replica, S.READY)
                            logger.info(
                                "%s replica %s READY from %s",
                                "candidate" if self._update_blocks_topology() else "committed",
                                replica.replica_id,
                                replica.loaded_model_path,
                            )
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

            if self._update_blocks_topology() and not self.closing:
                if self._check_update_deadline():
                    return
                try:
                    self._reconcile_policy()
                except Exception as exc:
                    self._fail_update(exc)
                return

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
                "policy": asdict(self.committed_policy),
                "serving_policy_version": self.serving_policy_version,
                "bootstrap_snapshot": asdict(self.bootstrap_snapshot),
                "topology_limitation": (
                    "fixed topology; no stale bootstrap replacement or catch-up"
                    if self.spec.replica.weight_sync == "hccl"
                    else None
                ),
                "transfer_session": self.transfer.status(),
                "transport": self.transfer.transport_status(),
                "policy_update": asdict(self.policy_update) if self.policy_update else None,
                "accepting_routes": self.traffic.accepting_routes,
                "serving_ready_for_external_update": (
                    self.spec.replica.weight_sync == "hccl"
                    and not self.closing
                    and not self.generation_paused
                    and self.traffic.accepting_routes
                    and not (
                        self.transfer.transport and self.transfer.transport.state == "POISONED"
                    )
                    and self._stable_for_policy_update(replicas)
                ),
                "generation_paused": self.generation_paused,
                "route_pause_reasons": self.traffic.route_pause_reasons,
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
            return manager.set_desired(count)

    def begin_policy_update(
        self, deployment_id: str, snapshot_path: str | None = None, **kwargs
    ) -> dict:
        with self._lock:
            if self._closed:
                raise PolicyConflict("controller is closing")
            return self.deployments[deployment_id].begin_policy_update(snapshot_path, **kwargs)

    def _generation_manager(self, deployment_id: str) -> ReplicaManager:
        with self._lock:
            if self._closed:
                raise PolicyConflict("controller is closing")
            return self.deployments[deployment_id]

    def pause_generation(self, deployment_id: str) -> dict:
        return self._generation_manager(deployment_id).pause_generation()

    def resume_generation(self, deployment_id: str) -> dict:
        return self._generation_manager(deployment_id).resume_generation()

    def get_policy_update(self, deployment_id: str, update_id: str) -> dict:
        return self.deployments[deployment_id].get_policy_update(update_id)

    def commit_policy_update(self, deployment_id: str, update_id: str, policy_version: int) -> dict:
        with self._lock:
            if self._closed:
                raise PolicyConflict("controller is closing")
            return self.deployments[deployment_id].commit_policy_update(update_id, policy_version)

    def reconcile(self) -> None:
        with self._lock:
            managers = list(self.deployments.values())
        for manager in managers:
            manager.reconcile()

    def close(self, timeout: float = 600) -> None:
        with self._lock:
            self._closed = True
            for manager in self.deployments.values():
                with manager.generation_lock, manager.mutation_lock, manager.traffic.lock:
                    if manager._update_blocks_topology():
                        manager._fail_update(RuntimeError("controller shutting down"))
                    manager.closing = True
                    manager.traffic.pause_routes("shutdown")
                    manager.desired = 0
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
