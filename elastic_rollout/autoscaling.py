"""Framework-neutral pressure policy and standalone polling runner.

Missing demand or engine samples never count as evidence for scale-in.
"""

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class AutoscalingConfig:
    policy_mode: str = "service"
    queue_high: float = 10.0
    wait_high_s: float = 5.0
    kv_high: float = 0.85
    kv_low: float = 0.30
    projected_kv_limit: float = 0.50
    scale_out_persistence_s: float = 30.0
    scale_in_persistence_s: float = 120.0
    scale_out_cooldown_s: float = 60.0
    scale_in_cooldown_s: float = 300.0
    sample_ttl_s: float = 30.0
    window_s: float = 60.0
    ttft_high_s: float = 10.0
    max_scale_out_delta: int = 4
    throughput_variance_limit: float = 0.1
    pipeline_enabled: bool = False
    wait_high: float = 0.10
    wait_low: float = 0.05
    high_reports: int = 2
    trial_timeout_s: float = 900.0
    trial_min_cycles: int = 3
    trial_policy_advances: int = 2
    ongoing_wait_limit_s: float = 5.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "policy_mode":
                if value not in ("pipeline", "service"):
                    raise ValueError("policy_mode must be pipeline or service")
            elif name == "pipeline_enabled":
                if type(value) is not bool:
                    raise ValueError("pipeline_enabled must be boolean")
            elif isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.kv_low < self.projected_kv_limit < self.kv_high < 1:
            raise ValueError("invalid KV thresholds")
        if not 0 < self.wait_low < self.wait_high < 1 or type(self.high_reports) is not int:
            raise ValueError("invalid pipeline thresholds/report count")
        if any(
            type(getattr(self, key)) is not int
            for key in ("trial_min_cycles", "trial_policy_advances")
        ):
            raise ValueError("trial cycle/version counts must be integers")
        if type(self.max_scale_out_delta) is not int:
            raise ValueError("max_scale_out_delta must be an integer")


@dataclass(frozen=True)
class WorkloadSignal:
    observed_at: float
    window_s: float
    phase: str
    trainer_wait_fraction: float
    produced: float
    consumed: float

    def valid(self, now, ttl):
        values = (
            self.observed_at,
            self.window_s,
            self.trainer_wait_fraction,
            self.produced,
            self.consumed,
        )
        return (
            all(
                not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                for v in values
            )
            and 0 <= now - self.observed_at <= ttl
            and self.window_s > 0
            and self.phase == "rollout"
            and 0 <= self.trainer_wait_fraction <= 1
            and self.produced >= 0
            and self.consumed >= 0
        )


class ServicePressurePolicy:
    """Relax-style engine pressure rules; no training inventory shrink trial.

    Reference: redai-studio/Relax 6a2ad57, scaling_decision.py.
    Adapted to vLLM metrics and ElasticRollout's lifecycle/coverage contract.
    """

    def __init__(self, config=None, *, interval=5.0):
        self.config = config or AutoscalingConfig()
        self.interval = interval
        self.high_since = self.low_since = None
        self.last_out = self.last_in = -math.inf
        self.samples = []
        self.last_observation = None
        self.identity = None
        self.streaks = {}

    def decide(self, status, metrics, workload=None, *, now=None):
        now = time.time() if now is None else now
        c, s = self.config, status["scaling"]
        ready = status["ready_replicas"]

        def hold(reason):
            return {"action": "HOLD", "target": status["desired_replicas"], "reason": reason}

        if (
            status["generation_paused"]
            or not status["accepting_routes"]
            or s["transaction_owned"]
            or s["retirements"]
            or s["pending_replicas"]
            or s["failures"] >= s["retry_limit"]
            or ready < s["minimum"]
            or ready != status["desired_replicas"]
        ):
            self.streaks.clear()
            self.samples.clear()
            self.high_since = self.low_since = None
            return hold("lifecycle or retry gate")
        identity = (s.get("revision"), ready)
        timestamp = metrics.get("observed_at")
        if (
            identity != self.identity
            or self.last_observation is not None
            and now - self.last_observation > min(c.sample_ttl_s, 2 * self.interval)
        ):
            self.streaks.clear()
            self.samples.clear()
        self.identity = identity

        def number(value):
            return type(value) in (int, float) and math.isfinite(value) and value >= 0

        if not number(timestamp) or not 0 <= now - timestamp <= c.sample_ttl_s:
            self.streaks.clear()
            self.samples.clear()
            self.high_since = self.low_since = None
            return hold("stale or missing engine metrics")
        if self.last_observation is not None and timestamp <= self.last_observation:
            return hold("waiting for a new engine observation")
        self.last_observation = timestamp
        values = {
            k: metrics[k]
            for k in ("queue_per_replica", "kv_usage", "wait_p95_s", "ttft_p95_s", "generation_tps")
            if number(metrics.get(k))
        }
        if values.get("kv_usage", 0) > 1:
            values.pop("kv_usage")
        self.samples.append((timestamp, values))
        self.samples = [(t, v) for t, v in self.samples if timestamp - t <= c.window_s][-10:]
        throughputs = [v["generation_tps"] for _, v in self.samples if "generation_tps" in v]
        mean = sum(throughputs) / len(throughputs) if throughputs else 0
        variance = max((abs(v - mean) for v in throughputs), default=0) / mean if mean else 0
        queue, kv = values.get("queue_per_replica"), values.get("kv_usage")
        coverage = metrics.get("coverage", 0)
        conditions = {
            "kv_high": kv is not None and kv > c.kv_high,
            "queue_high": queue is not None and queue > c.queue_high,
            "latency_high": values.get("wait_p95_s", 0) > c.wait_high_s
            or metrics.get("queue_p95_overflow") is True,
            "ttft_high": values.get("ttft_p95_s", 0) > c.ttft_high_s
            or metrics.get("ttft_p95_overflow") is True,
            "low": coverage == 1
            and queue == 0
            and kv is not None
            and kv < c.kv_low
            and len(throughputs) >= 3
            and len(throughputs) == len(self.samples)
            and variance < c.throughput_variance_limit,
        }
        for key, active in conditions.items():
            if not active or not number(coverage) or coverage < (1 if key == "low" else 0.5):
                self.streaks.pop(key, None)
            else:
                self.streaks.setdefault(key, timestamp)
        self.high_since = min((v for k, v in self.streaks.items() if k != "low"), default=None)
        self.low_since = self.streaks.get("low")
        # Relax applies a cooldown based on the previous scale direction.
        last = max(self.last_out, self.last_in)
        cooldown = (
            c.scale_out_cooldown_s if self.last_out >= self.last_in else c.scale_in_cooldown_s
        )
        if now - last < cooldown:
            return hold("scale operation cooldown")
        if any(
            timestamp - start >= c.scale_out_persistence_s
            for key, start in self.streaks.items()
            if key != "low"
        ):
            usage_delta = int((kv - 0.7) / 0.1) if kv is not None and kv > 0.9 else 1
            queue_delta = (
                max(1, int((queue * ready - ready * c.queue_high) // (c.queue_high * 2)))
                if queue is not None
                else 1
            )
            target = min(
                s["maximum"], ready + min(c.max_scale_out_delta, max(usage_delta, queue_delta, 1))
            )
            if target > ready:
                return {
                    "action": "OUT",
                    "target": target,
                    "reason": "Relax sustained engine pressure",
                }
        if (
            self.low_since is not None
            and timestamp - self.low_since >= c.scale_in_persistence_s
            and ready > s["minimum"]
            and kv * ready / (ready - 1) <= c.projected_kv_limit
        ):
            return {
                "action": "IN",
                "target": ready - 1,
                "reason": "Relax sustained low usage, empty queue and stable throughput",
            }
        return hold("engine pressure persistence, bounds or incomplete shrink evidence")

    def applied(self, decision, now):
        if decision["action"] == "OUT":
            self.last_out = now
        elif decision["action"] == "IN":
            self.last_in = now
        self.streaks.clear()
        self.samples.clear()
        self.high_since = self.low_since = None


class PipelineBalancePolicy:
    @staticmethod
    def scale_out_target(ready, signal):
        ratio_target = math.ceil(ready / max(1 - signal.trainer_wait_fraction, 1e-9))
        return min(ready + 2, max(ready + 1, ratio_target))


class TrainingPipelinePolicy:
    """Wait-driven expansion and reversible shrink trials; legacy signals vetoed."""

    def __init__(self, config=None):
        self.config = config or AutoscalingConfig()
        self.identity = None
        self.sequence = -1
        self.high_count = 0
        self.high_since = self.low_since = None
        self.last_out = self.last_in = -math.inf
        self.last_report_at = None
        self.low_start_cycles = None

    def decide(self, status, metrics, workload=None, *, now=None):
        from .pipeline import snapshot_from_body

        now = time.time() if now is None else now
        s = status["scaling"]
        c = AutoscalingConfig(**s["policy_config"]) if "policy_config" in s else self.config
        trial = s.get("trial")

        def decision(action, reason, target=None):
            result = {
                "action": action,
                "reason": reason,
                "target": status["desired_replicas"] if target is None else target,
            }
            if trial:
                result["trial_id"] = trial["trial_id"]
            return result

        def invalid(reason):
            self.high_count = 0
            self.high_since = self.low_since = None
            self.low_start_cycles = None
            return decision("TRIAL_CANCEL" if trial else "HOLD", reason)

        if s["transaction_owned"]:
            return decision(
                "HOLD", "required weight update in progress; Core watchdog retains deadline"
            )
        try:
            p = snapshot_from_body(workload or {})
        except (TypeError, ValueError):
            return invalid("missing or invalid native pipeline measurements")
        if (
            p.phase == "sync"
            and 0 <= now - p.observed_at <= c.sample_ttl_s
            and p.topology_revision == s.get("revision")
            and (not trial or p.run_id == trial["run_id"])
        ):
            return decision("HOLD", "weight synchronization excluded from trial evidence")
        if (
            trial
            and p.elapsed == 0
            and p.phase == "training"
            and p.attribution_known
            and p.wait_unknown_s == 0
            and 0 <= now - p.observed_at <= c.sample_ttl_s
            and p.topology_revision == s.get("revision")
            and p.run_id == trial["run_id"]
            and p.unit == trial["unit"]
            and p.policy_version <= status["serving_policy_version"]
        ):
            if (
                p.ongoing_wait_s > c.ongoing_wait_limit_s
                or not p.eligibility_known
                or p.eligible_units < p.next_batch_units
            ):
                return decision("TRIAL_CANCEL", "starvation while collecting first trial cycle")
            return decision("HOLD", "collecting first complete cycle after admission revision")
        if not p.usable(now, c.sample_ttl_s, s.get("revision")):
            return invalid("stale, unknown or mismatched pipeline window")
        # Core validated the version when it accepted this report. A normal
        # weight commit cannot invalidate a still-fresh capacity measurement.
        if p.policy_version > status["serving_policy_version"]:
            return invalid("pipeline policy version is ahead of Core")
        identity = (p.run_id, p.topology_revision)
        if identity != self.identity:
            self.identity, self.sequence = identity, -1
            self.high_count = 0
            self.high_since = self.low_since = None
            self.low_start_cycles = None
        if p.sequence <= self.sequence:
            return decision("HOLD", "waiting for a new pipeline report")
        self.sequence = p.sequence
        if self.last_report_at is not None and p.observed_at - self.last_report_at > c.sample_ttl_s:
            self.high_count = 0
            self.high_since = self.low_since = None
            self.low_start_cycles = None
        self.last_report_at = p.observed_at
        if (
            status["generation_paused"]
            or not status["accepting_routes"]
            or s["transaction_owned"]
            or s["retirements"]
            or s["pending_replicas"]
            or s["failures"] >= s["retry_limit"]
        ):
            return invalid("lifecycle gate; do not infer capacity from paused work")
        ready = status["ready_replicas"]
        if trial:
            if p.run_id != trial["run_id"] or p.unit != trial["unit"]:
                return invalid("producer epoch/unit changed during trial")
            if p.total_fraction > c.wait_high or p.ongoing_wait_s > c.ongoing_wait_limit_s:
                return decision("TRIAL_CANCEL", "training starvation during shrink trial")
            if (
                not p.eligibility_known
                or p.next_batch_units <= 0
                or p.eligible_units < p.next_batch_units
            ):
                return decision("TRIAL_CANCEL", "next native batch lacks eligible inventory")
            if (
                p.total_fraction < c.wait_low
                and p.ongoing_wait_s == 0
                and (p.window_cycles is None or p.window_cycles >= c.trial_min_cycles)
                and p.completed_cycles >= trial["completed_cycles"] + c.trial_min_cycles
                and p.policy_version >= trial["policy_version"] + c.trial_policy_advances
                and p.consumed_total >= trial["consumed_total"] + trial["eligible_units"]
                and p.oldest_buffer_policy_version is not None
                and p.oldest_buffer_policy_version > trial["policy_version"]
                and p.oldest_consumed_policy_version is not None
                and p.oldest_consumed_policy_version > trial["policy_version"]
                and trial.get("inflight", 0) == 0
            ):
                return decision(
                    "TRIAL_COMMIT", "fresh inventory turnover and stable trainer wait", ready - 1
                )
            return decision("HOLD", "observing fresh production through shrink trial")
        high = (
            p.data_fraction > c.wait_high
            and p.available_work
            and p.dispatch_unconstrained
            and (p.window_cycles if p.window_cycles is not None else p.completed_cycles)
            >= c.trial_min_cycles
            and p.train_path_s > 0
        )
        self.high_count = self.high_count + 1 if high else 0
        self.high_since = (now if self.high_since is None else self.high_since) if high else None
        if (
            high
            and self.high_count >= c.high_reports
            and now - self.high_since >= c.scale_out_persistence_s
            and now - self.last_out >= c.scale_out_cooldown_s
        ):
            target = min(s["maximum"], ready + 2, math.ceil(ready / max(1 - p.data_fraction, 1e-9)))
            if target > max(status["desired_replicas"], ready + s["pending_replicas"]):
                return decision("OUT", "sustained attributed trainer data wait", target)
        low = (
            p.total_fraction < c.wait_low
            and p.ongoing_wait_s == 0
            and p.eligibility_known
            and p.next_batch_units > 0
            and p.eligible_units >= p.next_batch_units
            and (p.window_cycles if p.window_cycles is not None else p.completed_cycles)
            >= c.trial_min_cycles
            and p.oldest_buffer_policy_version is not None
            and p.oldest_consumed_policy_version is not None
        )
        if not low:
            self.low_start_cycles = None
        elif self.low_start_cycles is None:
            self.low_start_cycles = p.completed_cycles
        self.low_since = (now if self.low_since is None else self.low_since) if low else None
        if (
            low
            and ready > s["minimum"]
            and ready == status["desired_replicas"]
            and now - self.low_since >= c.scale_in_persistence_s
            and now - self.last_in >= c.scale_in_cooldown_s
            and now >= s.get("scale_in_after", 0)
            and p.completed_cycles >= self.low_start_cycles + c.trial_min_cycles
        ):
            return decision("TRIAL_START", "sustained low trainer wait; test remaining capacity")
        return decision("HOLD", "dead band, persistence or incomplete shrink evidence")

    def applied(self, decision, now):
        if decision["action"] == "OUT":
            self.last_out = now
        elif decision["action"] in ("TRIAL_COMMIT", "TRIAL_CANCEL"):
            self.last_in = now
        self.high_count = 0
        self.high_since = self.low_since = None


def run_autoscaler(client, config, *, once=False, activate=False, interval=5.0):
    import json

    policy = (
        TrainingPipelinePolicy(config)
        if config.policy_mode == "pipeline"
        else ServicePressurePolicy(config, interval=interval)
    )
    if activate:
        from dataclasses import asdict

        client.scaling({"mode": "AUTO", "policy_config": asdict(config)})
    while True:
        snapshot = client.scaling()
        now = time.time()
        decision = policy.decide(
            snapshot["status"], snapshot["metrics"], snapshot["workload"], now=now
        )
        if (
            activate
            and decision["action"] != "HOLD"
            and snapshot["status"]["scaling"]["mode"] == "AUTO"
        ):
            body = {
                "desired_replicas": decision["target"],
                "revision": snapshot["status"]["scaling"]["revision"],
            }
            if decision["action"].startswith("TRIAL_"):
                body.pop("desired_replicas")
                body.update(trial_action=decision["action"], trial_id=decision.get("trial_id"))
            from .client import ControlError

            try:
                client.scaling(body)
            except ControlError as exc:
                if exc.status != 409:
                    raise
                decision = {
                    "action": "HOLD",
                    "reason": "Core revision or lifecycle changed; resample",
                }
                policy = (
                    TrainingPipelinePolicy(config)
                    if config.policy_mode == "pipeline"
                    else ServicePressurePolicy(config, interval=interval)
                )
            else:
                policy.applied(decision, now)
        print(json.dumps(decision), flush=True)
        if once:
            return
        time.sleep(interval)
