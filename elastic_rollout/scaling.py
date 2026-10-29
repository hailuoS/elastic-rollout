"""Elastic capacity intent, metrics and shrink-trial controls for ReplicaManager."""

from dataclasses import asdict

from .state import STARTUP, TERMINAL, PolicyConflict, S


class ScalingControl:
    def _set_elastic_desired(self, count, *, automatic=False, revision=None):
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

    def _scaling_status(self, result):
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
