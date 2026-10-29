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
    nonnegative_int,
)
from .api import (
    PolicyPreparationKind as K,
)
from .resource import NoCapacity, StaticResourceProvider
from .scaling import ScalingControl
from .state import (
    DRAIN,
    STARTUP,
    TERMINAL,
    ElasticTopologyNotReady,
    GenerationControlTimeout,
    PolicyConflict,
    S,
    U,
)
from .traffic import TrafficManager
from .weight_transfer import RECOVERY_REQUIRED, WeightTransferControl

logger = logging.getLogger(__name__)


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


class ReplicaManager(ScalingControl):
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
        from .multi_transfer import MultiTransferCoordinator
        from .weight_transfer import TransferCoordinator

        self.transfer = (
            MultiTransferCoordinator(self) if spec.elastic_scaling else TransferCoordinator(self)
        )
        self.policy_update: PolicyUpdateDescriptor | None = None
        self.policy_updates: dict[str, PolicyUpdateDescriptor] = {}
        self.closing = False

        self.base_count = spec.desired_replicas
        self.base_ids = []
        self.transaction_owned = False
        self.scaling_mode = "MANUAL"
        self.scaling_revision = 0
        self.scale_failures = 0
        self.scale_in_after = 0.0
        self.policy_priority = False
        self.workload_signal = None
        self.pipeline_signal = None
        self.pipeline_run = None
        self.trial = None
        self.last_trial = None
        from .autoscaling import AutoscalingConfig

        self.policy_config = AutoscalingConfig()

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
        return bool(getattr(self, "transaction_owned", False)) or (
            self.policy_update is not None and self.policy_update.state != U.COMMITTED
        )

    def _stable_for_policy_update(self, replicas: list) -> bool:
        """The topology predicate shared by begin and the serving readiness view.

        Call with traffic.lock held. Begin additionally serializes mutations and
        checks the requested update kind and target version.
        """
        if self.spec.elastic_scaling:
            return (
                not self._update_blocks_topology()
                and not self.transfer.poisoned
                and len(self.base_ids) == self.base_count
                and all(self.traffic.replicas[rid].state == S.READY for rid in self.base_ids)
                and not any(r.state in (S.DRAINING, S.RETIRE_PENDING, S.STOPPING) for r in replicas)
                and not any(r.state == S.FAILED and r.resource for r in replicas)
            )

        return (
            not self._update_blocks_topology()
            and self.desired > 0
            and not any(r.state in STARTUP + DRAIN for r in replicas)
            and not any(r.state in TERMINAL and r.resource is not None for r in replicas)
            and sum(r.state == S.READY for r in replicas) == self.desired
        )

    def set_desired(self, count: int, *, automatic=False, revision=None) -> dict:
        if self.spec.elastic_scaling:
            return self._set_elastic_desired(count, automatic=automatic, revision=revision)
        # Preserve fixed-topology desired-state changes during normal reconciliation.
        # Begin/commit also hold this lock, so the policy gate is atomic here.
        with self.traffic.lock:
            if self.spec.replica.weight_sync in ("hccl", "nccl") and count != self.desired:
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
                if (
                    not paused
                    and self._external_update()
                    and self.policy_update.state != U.COMMITTED
                ):
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
                    r
                    for r in self.traffic.replicas.values()
                    if r.state in (S.READY, S.DRAINING) or (paused and r.state == S.JOIN_PENDING)
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
                    if hasattr(self, "transaction_owned"):
                        self.transaction_owned = False
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
        return bool(
            self.policy_update
            and self.policy_update.preparation_kind in (K.EXTERNAL_HCCL, K.EXTERNAL_NCCL)
        )

    def begin_policy_update(
        self,
        snapshot_path: str | None = None,
        preparation_kind: str = K.SNAPSHOT_RESTART,
        target_version: int | None = None,
    ) -> dict:
        kind = K(preparation_kind)
        if self.spec.elastic_scaling and kind == K.SNAPSHOT_RESTART:
            raise ValueError("elastic scaling requires external collective updates")
        if kind in (K.EXTERNAL_HCCL, K.EXTERNAL_NCCL):
            nonnegative_int(target_version, "target_version")
            if kind.value != "EXTERNAL_" + self.spec.replica.weight_sync.upper():
                raise PolicyConflict("deployment transport does not match external backend")
            if snapshot_path is not None:
                raise ValueError("EXTERNAL_HCCL has no snapshot_path")
        elif not isinstance(snapshot_path, str) or not snapshot_path.strip():
            raise ValueError("snapshot_path must be a non-empty string")
        with self._control_mutation():
            if (
                kind in (K.EXTERNAL_HCCL, K.EXTERNAL_NCCL)
                and self.transfer.transport
                and self.transfer.transport.state == "POISONED"
            ):
                raise PolicyConflict(RECOVERY_REQUIRED)
            if (
                kind in (K.EXTERNAL_HCCL, K.EXTERNAL_NCCL)
                and target_version <= self.serving_policy_version
            ):
                raise ValueError("target_version must exceed serving_policy_version")
            replicas = list(self.traffic.replicas.values())
            if self._update_blocks_topology():
                raise PolicyConflict("policy update active or failed")
            if self.desired <= 0:
                raise PolicyConflict("policy update requires at least one READY replica")
            if not self._stable_for_policy_update(replicas):
                if (
                    self.spec.elastic_scaling
                    and not self.transfer.poisoned
                    and not any(r.state == S.FAILED and (r.is_base or r.resource) for r in replicas)
                    and any(r.state in STARTUP + DRAIN + (S.RETIRE_PENDING,) for r in replicas)
                ):
                    raise ElasticTopologyNotReady(
                        "elastic topology changing; BEGIN created no transaction"
                    )
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
            if kind in (K.EXTERNAL_HCCL, K.EXTERNAL_NCCL):
                self.transfer.session = None
            if self.spec.elastic_scaling:
                update.optional_replica_ids = tuple(
                    r.replica_id
                    for r in self.traffic.replicas.values()
                    if r.state == S.JOIN_PENDING and not r.cancel_requested
                )
                update.transport_plan = [
                    {"transport_key": "base", "replica_ids": list(self.base_ids), "optional": False}
                ]
                for rid in update.target_replica_ids + update.optional_replica_ids:
                    if rid not in self.base_ids:
                        update.transport_plan.append(
                            {
                                "transport_key": rid,
                                "replica_ids": [rid],
                                "optional": rid in update.optional_replica_ids,
                            }
                        )
                self.transaction_owned = True
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
                if self.spec.elastic_scaling:
                    live = [self.traffic.replicas[rid] for rid in update.target_replica_ids]
                    if any(
                        key in self.transfer.children for key in self.transfer.optional_failures
                    ):
                        raise PolicyConflict(
                            "failed optional sender must acknowledge isolation before commit"
                        )
                    if any(
                        r.state != S.READY or r.loaded_policy_version != policy_version
                        for r in live
                    ):
                        raise PolicyConflict("required target has not loaded policy")
                    live += [
                        r
                        for rid in update.optional_replica_ids
                        if (r := self.traffic.replicas[rid]).state == S.JOIN_PENDING
                        and not r.cancel_requested
                        and r.prepared_update_id == update_id
                        and r.prepared_policy_version == policy_version
                    ]
                elif set(r.replica_id for r in live) != set(update.target_replica_ids) or any(
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
                if replica.state == S.JOIN_PENDING:
                    self._transition(replica, S.READY)
                replica.serving_policy_version = policy_version
                replica.loaded_policy_version = policy_version
            self.policy_priority = False
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

    def _transition(self, replica: ReplicaDescriptor, state: S) -> None:
        with self.traffic.lock:
            if state in (S.STARTING, S.READY):
                self._generation_pause_verified = False
            replica.transition(state)

    def reconcile(self):
        with self.generation_lock, self.mutation_lock:
            self.traffic.expire()
            if self.spec.elastic_scaling:
                with self.traffic.lock:
                    if self.trial and (
                        self.closing
                        or self.clock() >= self.trial["deadline_at"]
                        or not self.pipeline_signal
                        or self.clock() - self.pipeline_signal["observed_at"]
                        > self.policy_config.sample_ttl_s
                    ):
                        self._cancel_trial("trial deadline or reporter heartbeat lost")
                self.transfer.maintenance()
                if self._check_update_deadline() and not self.closing:
                    return
                if self.transaction_owned and not self.closing:
                    return
            elif self._update_blocks_topology() and not self.closing:
                if self._check_update_deadline():
                    return
                if self.policy_update.state == U.REQUESTED and not self._external_update():
                    self._update_transition(U.QUIESCING)
            for replica in list(self.traffic.replicas.values()):
                if not self.spec.elastic_scaling and self._check_update_deadline():
                    return
                try:
                    self._reconcile_replica(replica)
                except Exception as exc:
                    self._fail(replica, exc)
            if self.spec.elastic_scaling:
                self._reconcile_elastic_capacity()
            else:
                self._reconcile_fixed_capacity()

    def _cleanup(self, replica: ReplicaDescriptor) -> None:
        if self.spec.elastic_scaling and not replica.is_base and not self.closing:
            if self.transaction_owned:
                return
            if replica.state != S.STOPPING and self.transfer.request_retirement(replica):
                return
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
        if self.spec.elastic_scaling and not replica.is_base and replica.state not in TERMINAL:
            self.scale_failures += 1
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
        elif (
            self.serving_policy_version != self.bootstrap_snapshot.version
            and not self.spec.elastic_scaling
        ):
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
        if self.spec.elastic_scaling and len(self.base_ids) < self.base_count:
            replica.is_base = True
            self.base_ids.append(replica.replica_id)
        return True

    def status(self) -> dict:
        with self.traffic.lock:
            self.traffic.expire()
            replicas = list(self.traffic.replicas.values())
            result = {
                "spec": asdict(self.spec),
                "policy": asdict(self.committed_policy),
                "serving_policy_version": self.serving_policy_version,
                "bootstrap_snapshot": asdict(self.bootstrap_snapshot),
                "topology_limitation": (
                    "fixed topology; no stale bootstrap replacement or catch-up"
                    if self.spec.replica.weight_sync in ("hccl", "nccl")
                    else None
                ),
                "transfer_session": self.transfer.status(),
                "transport": self.transfer.transport_status(),
                "policy_update": asdict(self.policy_update) if self.policy_update else None,
                "accepting_routes": self.traffic.accepting_routes,
                "serving_ready_for_external_update": (
                    self.spec.replica.weight_sync in ("hccl", "nccl")
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

            return self._scaling_status(result) if self.spec.elastic_scaling else result

    def _reconcile_replica(self, r):
        elastic = self.spec.elastic_scaling
        if r.state == S.TERMINATED:
            return
        if r.state in (S.FAILED, S.STOPPING):
            if r.resource or r.state == S.STOPPING:
                self._cleanup(r)
            return
        if r.state == S.RETIRE_PENDING:
            if self.closing:
                self._transition(r, S.STOPPING)
                self._cleanup(r)
            return
        if self.closing and r.state in STARTUP and not elastic:
            self._transition(r, S.STOPPING)
            self._cleanup(r)
            return
        if not self.driver.alive(r):
            raise RuntimeError(
                "replica process exited; no stale replacement"
                if elastic
                else "replica process exited"
            )
        if elastic and (r.cancel_requested or self.closing):
            self.traffic.forget_replica(r.replica_id)
            if r.state in STARTUP or (
                not self.traffic.inflight.get(r.replica_id, 0) and self.driver.idle(r)
            ):
                self._cleanup(r)
            return
        if r.state in STARTUP:
            if self.clock() - r.started_at >= self.spec.replica.startup_timeout:
                raise TimeoutError("replica startup/health timeout")
            if not self.driver.health(r):
                return
            if not elastic and r.state == S.STARTING:
                self._transition(r, S.SYNCING)
                self._transition(r, S.WARMING)
            if self.driver.warmup(r, self._effective_spec(r.loaded_model_path)):
                if elastic and not r.is_base:
                    self.driver.pause_generation(r, timeout=30)
                    self._transition(r, S.JOIN_PENDING)
                else:
                    self._transition(r, S.READY)
        elif r.state in (S.READY, S.JOIN_PENDING, S.DRAINING):
            r.health_failures = 0 if self.driver.health(r) else r.health_failures + 1
            if r.health_failures >= 3:
                raise RuntimeError("replica failed three consecutive health probes")
        if r.state != S.DRAINING:
            return
        if elastic:
            if not self.traffic.inflight.get(r.replica_id, 0) and self.driver.idle(r):
                self._cleanup(r)
            elif self.clock() >= r.drain_deadline:
                self._transition(r, S.READY)
                r.error = "drain timeout; restored before retirement"
                r.admission_disabled = False
                self.scale_in_after = self.clock() + self.policy_config.scale_in_cooldown_s
                self.policy_priority = True
        elif self.clock() >= r.drain_deadline:
            self.traffic.forget_replica(r.replica_id)
            if not self.traffic.inflight.get(r.replica_id, 0) and self.driver.idle(r):
                self._transition(r, S.QUIESCED)
                self._transition(r, S.STOPPING)
                self._cleanup(r)

    def _reconcile_fixed_capacity(self):
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
                victims = sorted(ready, key=lambda r: self.traffic.inflight.get(r.replica_id, 0))
                for replica in victims[:excess]:
                    replica.drain_deadline = self.clock() + self.spec.drain_grace_period
                    replica.transition(S.DRAINING)
                return
        for _ in range(max(0, desired - len(ready))):
            if not self._create():
                break

    def _reconcile_elastic_capacity(self):
        if self.closing or self.transfer.poisoned:
            return
        # Never repair a lost member of the immutable base cohort.
        if any(self.traffic.replicas[rid].state == S.FAILED for rid in self.base_ids):
            self.traffic.pause_routes("base_failed")
            return
        live = [
            r
            for r in self.traffic.replicas.values()
            if r.state not in TERMINAL and not r.cancel_requested
        ]
        if any(r.state in (S.DRAINING, S.RETIRE_PENDING, S.STOPPING) for r in live):
            return
        if (
            len(live) > self.desired
            and self.clock() >= self.scale_in_after
            and not self.policy_priority
        ):
            victims = sorted(
                [r for r in live if not r.is_base and r.state == S.READY],
                key=lambda r: self.traffic.inflight.get(r.replica_id, 0),
            )
            for r in victims[: len(live) - self.desired]:
                with self.traffic.lock:
                    r.drain_deadline = self.clock() + self.spec.drain_grace_period
                    self._transition(r, S.DRAINING)
                    self.traffic.forget_replica(r.replica_id)
        elif len(live) < self.desired and self.scale_failures < self.spec.scale_retry_limit:
            for _ in range(self.desired - len(live)):
                if not self._create():
                    break


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
                # Stop new leases before waiting for an in-flight generation operation.
                with manager.traffic.lock:
                    manager.closing = True
                    manager.traffic.pause_routes("shutdown")
            for manager in self.deployments.values():
                with manager.generation_lock, manager.mutation_lock, manager.traffic.lock:
                    if manager._update_blocks_topology():
                        manager._fail_update(RuntimeError("controller shutting down"))
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
