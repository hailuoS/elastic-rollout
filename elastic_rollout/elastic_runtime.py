"""Opt-in MVP-4 replica lifecycle with trainer-owned communicator retirement."""

from dataclasses import asdict

from .api import PolicyPreparationKind as K
from .api import ReplicaState as S
from .runtime import STARTUP, TERMINAL, PolicyConflict, ReplicaManager, U


class ElasticReplicaManager(ReplicaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .multi_transfer import MultiTransferCoordinator

        self.transfer = MultiTransferCoordinator(self)
        self.base_count = self.spec.desired_replicas
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

    def set_desired(self, count, *, automatic=False, revision=None):
        from .api import nonnegative_int

        nonnegative_int(count, "desired_replicas")
        with self.traffic.lock:
            if self.closing or self.transfer.poisoned:
                raise PolicyConflict("deployment closing or transport poisoned")
            if not self.base_count <= count <= len(self.resources.slots):
                raise ValueError("desired must fit base cohort and static slot pool")
            if automatic:
                if self.scaling_mode != "AUTO" or revision != self.scaling_revision:
                    raise PolicyConflict("automatic scaling revision/mode changed")
                if self.scale_failures >= self.spec.scale_retry_limit:
                    raise PolicyConflict("scale retry budget exhausted; manual recovery required")
                if count < self.desired:
                    raise PolicyConflict("automatic scale-in requires a successful shrink trial")
            else:
                self._cancel_trial("manual desired override")
                self.scaling_mode = "MANUAL"
                self.scale_failures = 0
            self.desired = count
            self.scaling_revision += 1
            # Intent can change while frozen; physical cleanup waits for resume.
            pending = [
                r
                for r in self.traffic.replicas.values()
                if not r.is_base
                and not r.cancel_requested
                and r.state in STARTUP + (S.JOIN_PENDING,)
            ]
            active = sum(r.state == S.READY for r in self.traffic.replicas.values())
            for replica in pending[max(0, count - active) :]:
                replica.cancel_requested = True
            return self.status()

    def _stable_for_policy_update(self, replicas):
        return (
            not self._update_blocks_topology()
            and not self.transfer.poisoned
            and len(self.base_ids) == self.base_count
            and all(self.traffic.replicas[rid].state == S.READY for rid in self.base_ids)
            and not any(r.state in (S.DRAINING, S.RETIRE_PENDING, S.STOPPING) for r in replicas)
            and not any(r.state == S.FAILED and r.resource for r in replicas)
        )

    def begin_policy_update(
        self, snapshot_path=None, preparation_kind=K.SNAPSHOT_RESTART, target_version=None
    ):
        if preparation_kind not in (K.EXTERNAL_HCCL, K.EXTERNAL_NCCL):
            raise ValueError("elastic scaling requires external collective updates")
        # Super takes the lifecycle lock, so freeze construction is atomic with
        # reconcile and desired intents under the reentrant routing lock.
        with self.traffic.lock:
            super().begin_policy_update(snapshot_path, preparation_kind, target_version)
            u = self.policy_update
            u.optional_replica_ids = tuple(
                r.replica_id
                for r in self.traffic.replicas.values()
                if r.state == S.JOIN_PENDING and not r.cancel_requested
            )
            u.transport_plan = [
                {"transport_key": "base", "replica_ids": list(self.base_ids), "optional": False}
            ]
            for rid in u.target_replica_ids + u.optional_replica_ids:
                if rid not in self.base_ids:
                    u.transport_plan.append(
                        {
                            "transport_key": rid,
                            "replica_ids": [rid],
                            "optional": rid in u.optional_replica_ids,
                        }
                    )
            self.transaction_owned = True
            return asdict(u)

    def commit_policy_update(self, update_id, policy_version):
        # The legacy commit verifies and advances every live replica. Isolate the
        # frozen required set; pending candidates retain their physical version.
        with self._control_mutation():
            u = self.policy_updates[update_id]
            if u.state == U.COMMITTED and u.committed_version == policy_version:
                return asdict(u)
            if (
                u is not self.policy_update
                or self._check_update_deadline()
                or u.state != U.PREPARED
            ):
                raise PolicyConflict("policy update must be PREPARED before commit")
            if type(policy_version) is not int or policy_version != u.target_version:
                raise ValueError("commit version must equal target_version")
            required = [self.traffic.replicas[rid] for rid in u.target_replica_ids]
            if any(key in self.transfer.children for key in self.transfer.optional_failures):
                raise PolicyConflict(
                    "failed optional sender must acknowledge isolation before commit"
                )
            if any(
                r.state != S.READY or r.loaded_policy_version != policy_version for r in required
            ):
                raise PolicyConflict("required target has not loaded policy")
            admitted = [
                self.traffic.replicas[rid]
                for rid in u.optional_replica_ids
                if self.traffic.replicas[rid].state == S.JOIN_PENDING
                and not self.traffic.replicas[rid].cancel_requested
                and self.traffic.replicas[rid].prepared_update_id == update_id
                and self.traffic.replicas[rid].prepared_policy_version == policy_version
            ]
            from .api import PolicySnapshot

            for replica in required + admitted:
                replica.loaded_policy_version = replica.serving_policy_version = policy_version
                if replica in admitted:
                    self._transition(replica, S.READY)
            self.serving_policy_version = policy_version
            self.committed_policy = PolicySnapshot(
                policy_version, self.bootstrap_snapshot.model_path
            )
            self.traffic.sticky.clear()
            u.committed_version = policy_version
            self._update_transition(U.COMMITTED)
            self.traffic.resume_routes("policy_update")
            self.policy_priority = False
            return asdict(u)

    def _create(self):
        before = set(self.traffic.replicas)
        result = super()._create()
        for rid in set(self.traffic.replicas) - before:
            replica = self.traffic.replicas[rid]
            if len(self.base_ids) < self.base_count:
                replica.is_base = True
                self.base_ids.append(rid)
        return result

    def _cleanup(self, replica):
        if not replica.is_base and not self.closing:
            if self.transaction_owned:
                return
            if replica.state != S.STOPPING and self.transfer.request_retirement(replica):
                return
        super()._cleanup(replica)

    def _fail(self, replica, error):
        if not replica.is_base and replica.state not in TERMINAL:
            self.scale_failures += 1
        super()._fail(replica, error)

    def reconcile(self):
        with self.generation_lock, self.mutation_lock:
            self.traffic.expire()
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
            # Physical state belongs to the transaction through resume completion.
            if self.transaction_owned and not self.closing:
                return
            for r in list(self.traffic.replicas.values()):
                try:
                    if r.state == S.TERMINATED:
                        continue
                    if r.state in (S.FAILED, S.STOPPING):
                        if r.resource:
                            self._cleanup(r)
                        continue
                    if r.state == S.RETIRE_PENDING:
                        if self.closing:
                            self._transition(r, S.STOPPING)
                            self._cleanup(r)
                        continue
                    if not self.driver.alive(r):
                        raise RuntimeError("replica process exited; no stale replacement")
                    if r.cancel_requested or self.closing:
                        self.traffic.forget_replica(r.replica_id)
                        if r.state in STARTUP or (
                            not self.traffic.inflight.get(r.replica_id, 0) and self.driver.idle(r)
                        ):
                            self._cleanup(r)
                        continue
                    if r.state in STARTUP:
                        if self.clock() - r.started_at >= self.spec.replica.startup_timeout:
                            raise TimeoutError("replica startup timeout")
                        if self.driver.health(r) and self.driver.warmup(
                            r, self._effective_spec(r.loaded_model_path)
                        ):
                            if r.is_base:
                                self._transition(r, S.READY)
                            else:
                                self.driver.pause_generation(r, timeout=30)
                                self._transition(r, S.JOIN_PENDING)
                    elif r.state in (S.READY, S.JOIN_PENDING, S.DRAINING):
                        r.health_failures = 0 if self.driver.health(r) else r.health_failures + 1
                        if r.health_failures >= 3:
                            raise RuntimeError("three consecutive failed health probes")
                    if r.state == S.DRAINING:
                        if not self.traffic.inflight.get(r.replica_id, 0) and self.driver.idle(r):
                            self._cleanup(r)
                        elif self.clock() >= r.drain_deadline:
                            self._transition(r, S.READY)
                            r.error = "drain timeout; restored before retirement"
                            r.admission_disabled = False
                            self.scale_in_after = (
                                self.clock() + self.policy_config.scale_in_cooldown_s
                            )
                            self.policy_priority = True
                except Exception as exc:
                    self._fail(r, exc)
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

    def status(self):
        with self.traffic.lock:
            return self._elastic_status()

    def _elastic_status(self):
        result = super().status()
        result["topology_limitation"] = "immutable base plus per-replica persistent transports"
        result["scaling"] = {
            "mode": self.scaling_mode,
            "revision": self.scaling_revision,
            "minimum": self.base_count,
            "maximum": len(self.resources.slots),
            "failures": self.scale_failures,
            "retry_limit": self.spec.scale_retry_limit,
            "transaction_owned": self.transaction_owned,
            "policy_safe_point": not self.closing
            and self._stable_for_policy_update(list(self.traffic.replicas.values())),
            "pending_replicas": sum(
                r.state in STARTUP + (S.JOIN_PENDING,) and not r.cancel_requested
                for r in self.traffic.replicas.values()
            ),
            "retirements": list(self.transfer.retirements.values()),
            "trial": {
                **self.trial,
                "inflight": self.traffic.inflight.get(self.trial["replica_id"], 0),
            }
            if self.trial
            else None,
            "last_trial": self.last_trial,
            "scale_in_after": self.scale_in_after,
            "policy_config": asdict(self.policy_config),
        }
        return result

    def scaling(self, body=None):
        if body is not None:
            from .autoscaling import AutoscalingConfig

            proposed_config = None
            if "policy_config" in body:
                if body.get("mode") != "AUTO":
                    raise ValueError("policy configuration must accompany AUTO activation")
                proposed_config = AutoscalingConfig(**body["policy_config"])
            if (
                sum(
                    key in body
                    for key in ("mode", "desired_replicas", "trial_action", "reset_pipeline")
                )
                > 1
            ):
                raise ValueError("one scaling operation per request required")
            with self.traffic.lock:
                if "mode" in body:
                    if body["mode"] not in ("AUTO", "MANUAL"):
                        raise ValueError("mode must be AUTO or MANUAL")
                    self.scaling_mode = body["mode"]
                    if proposed_config is not None:
                        self._cancel_trial("policy configuration changed")
                        self.policy_config = proposed_config
                    if body["mode"] == "MANUAL":
                        self._cancel_trial("manual mode")
                    self.scaling_revision += 1
                if body.get("reset_pipeline"):
                    if self.scaling_mode != "MANUAL":
                        raise PolicyConflict("pipeline reset requires MANUAL mode")
                    self._cancel_trial("pipeline epoch reset")
                    self.pipeline_run = self.pipeline_signal = None
                    self.scaling_revision += 1
                if "trial_action" in body:
                    self._trial_action(body)
                if "desired_replicas" in body:
                    if (
                        type(body["desired_replicas"]) is not int
                        or body["desired_replicas"] < self.desired
                    ):
                        raise PolicyConflict(
                            "automatic scale-in requires a successful shrink trial"
                        )
                    if self.scaling_mode != "AUTO" or body.get("revision") != self.scaling_revision:
                        raise PolicyConflict("automatic scaling revision/mode changed")
                    if self.trial:
                        self._cancel_trial("new capacity target")
                        body = dict(body, revision=self.scaling_revision)
                    self.set_desired(
                        body["desired_replicas"], automatic=True, revision=body.get("revision")
                    )
        with self.traffic.lock:
            status = self.status()
            revision = self.scaling_revision
            targets = [r for r in self.traffic.replicas.values() if r.state == S.READY]
            workload = self.pipeline_signal or self.workload_signal
        # Network scraping never holds the routing lock. Reject a changing
        # topology rather than combine observations from different incarnations.
        samples = []
        for replica in targets:
            try:
                samples.append(self.driver.pressure(replica))
            except Exception:
                samples.append({})
        metrics = {"observed_at": self.clock()}
        with self.traffic.lock:
            current = [r.replica_id for r in self.traffic.replicas.values() if r.state == S.READY]
            if (
                current == [r.replica_id for r in targets]
                and samples
                and revision == self.scaling_revision
                and not self.transaction_owned
                and self.traffic.accepting_routes
                and not self.generation_paused
            ):
                for key in ("queue_per_replica", "wait_p95_s", "kv_usage"):
                    if all(key in sample for sample in samples):
                        values = [sample[key] for sample in samples]
                        metrics[key] = (
                            sum(values) / len(values) if key == "queue_per_replica" else max(values)
                        )
        return {"status": status, "metrics": metrics, "workload": workload}

    def publish_pipeline(self, body):
        from .pipeline import snapshot_from_body

        p = snapshot_from_body(body)
        with self.traffic.lock:
            if not 0 <= self.clock() - p.observed_at <= self.policy_config.sample_ttl_s:
                raise ValueError("pipeline report stale or in the future")
            if p.topology_revision != self.scaling_revision:
                raise PolicyConflict("pipeline topology revision changed")
            if self.pipeline_run is not None and p.run_id != self.pipeline_run:
                raise PolicyConflict("pipeline producer epoch changed; reset in MANUAL mode")
            old = self.pipeline_signal
            if old and p.sequence <= old["sequence"]:
                if asdict(p) == old:
                    return {"accepted": True, "duplicate": True}
                raise PolicyConflict("pipeline report reordered or sequence reused")
            if old and (
                p.completed_cycles < old["completed_cycles"]
                or p.consumed_total < old["consumed_total"]
                or p.unit != old["unit"]
                or p.observed_at <= old["observed_at"]
            ):
                raise PolicyConflict("pipeline native counters/unit changed without new epoch")
            if p.policy_version != self.serving_policy_version:
                raise PolicyConflict("pipeline report policy version changed")
            self.pipeline_run, self.pipeline_signal = p.run_id, asdict(p)
            return {"accepted": True}

    def _cancel_trial(self, reason):
        if self.trial:
            replica = self.traffic.replicas[self.trial["replica_id"]]
            replica.admission_disabled = False
            self.last_trial = {**self.trial, "outcome": "CANCELLED", "reason": reason}
            self.trial = None
            self.scale_in_after = self.clock() + self.policy_config.scale_in_cooldown_s
            self.scaling_revision += 1

    def _trial_action(self, body):
        from uuid import uuid4

        from .pipeline import snapshot_from_body

        if self.scaling_mode != "AUTO" or body.get("revision") != self.scaling_revision:
            raise PolicyConflict("trial mode/revision changed")
        action = body["trial_action"]
        if action not in ("TRIAL_START", "TRIAL_CANCEL", "TRIAL_COMMIT"):
            raise ValueError("unknown trial action")
        if action != "TRIAL_START":
            if not self.trial or body.get("trial_id") != self.trial["trial_id"]:
                raise PolicyConflict("trial identity mismatch")
            if action == "TRIAL_CANCEL":
                self._cancel_trial("autoscaler restored admission")
                return
        if self.transaction_owned or self.generation_paused or not self.traffic.accepting_routes:
            raise PolicyConflict("policy transaction owns topology")
        p = snapshot_from_body(self.pipeline_signal or {})
        if (
            not p.usable(self.clock(), self.policy_config.sample_ttl_s, self.scaling_revision)
            or p.policy_version > self.serving_policy_version
            or p.total_fraction >= self.policy_config.wait_low
            or p.ongoing_wait_s
            or not p.eligibility_known
            or p.next_batch_units <= 0
            or p.eligible_units < p.next_batch_units
        ):
            raise PolicyConflict("trial requires current low-wait native batch evidence")
        if action == "TRIAL_START":
            live = [r for r in self.traffic.replicas.values() if r.state not in TERMINAL]
            candidates = [r for r in live if r.state == S.READY and not r.is_base]
            if (
                self.trial
                or not candidates
                or self.clock() < self.scale_in_after
                or any(r.state != S.READY for r in live)
                or len(live) != self.desired
                or self.transfer.poisoned
                or p.completed_cycles < self.policy_config.trial_min_cycles
                or p.oldest_buffer_policy_version is None
                or p.oldest_consumed_policy_version is None
            ):
                raise PolicyConflict("deployment not eligible for shrink trial")
            replica = min(candidates, key=lambda r: self.traffic.inflight.get(r.replica_id, 0))
            replica.admission_disabled = True
            self.traffic.forget_replica(replica.replica_id)
            self.trial = {
                "trial_id": uuid4().hex,
                "replica_id": replica.replica_id,
                "deadline_at": self.clock() + self.policy_config.trial_timeout_s,
                "run_id": p.run_id,
                "unit": p.unit,
                "completed_cycles": p.completed_cycles,
                # The cached report can predate a recent weight commit. The
                # turnover fence must start at admission change, never backdate.
                "policy_version": self.serving_policy_version,
                "consumed_total": p.consumed_total,
                "eligible_units": p.eligible_units,
            }
            self.scaling_revision += 1
            return
        t = self.trial
        r = self.traffic.replicas[t["replica_id"]]
        if (
            self.clock() >= t["deadline_at"]
            or p.run_id != t["run_id"]
            or p.unit != t["unit"]
            or p.completed_cycles < t["completed_cycles"] + self.policy_config.trial_min_cycles
            or (
                p.window_cycles is not None
                and p.window_cycles < self.policy_config.trial_min_cycles
            )
            or p.policy_version < t["policy_version"] + self.policy_config.trial_policy_advances
            or p.consumed_total < t["consumed_total"] + t["eligible_units"]
            or p.oldest_buffer_policy_version is None
            or p.oldest_buffer_policy_version <= t["policy_version"]
            or p.oldest_consumed_policy_version is None
            or p.oldest_consumed_policy_version <= t["policy_version"]
            or r.state != S.READY
            or self.traffic.inflight.get(r.replica_id, 0)
        ):
            raise PolicyConflict("trial has not proven fresh inventory turnover or drained leases")
        # Reconciler verifies physical idle before retirement, outside this lock.
        r.drain_deadline = self.clock() + self.spec.drain_grace_period
        self._transition(r, S.DRAINING)
        self.desired -= 1
        self.last_trial = {**t, "outcome": "COMMITTED"}
        self.trial = None
        self.scale_in_after = self.clock() + self.policy_config.scale_in_cooldown_s
        self.scaling_revision += 1
