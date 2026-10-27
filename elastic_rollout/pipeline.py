"""Validated, framework-neutral training-window metadata. No trajectories or tensors."""

import math
from dataclasses import asdict, dataclass
from threading import RLock


@dataclass(frozen=True)
class PipelineSnapshot:
    run_id: str
    sequence: int
    topology_revision: int
    observed_at: float
    window_s: float
    completed_cycles: int
    policy_version: int
    unit: str
    wait_data_s: float
    wait_staleness_s: float
    train_path_s: float
    wait_unknown_s: float = 0.0
    phase: str = "training"
    available_work: bool = False
    dispatch_unconstrained: bool = False
    attribution_known: bool = False
    eligible_units: int = 0
    next_batch_units: int = 0
    eligibility_known: bool = False
    oldest_buffer_policy_version: int | None = None
    oldest_consumed_policy_version: int | None = None
    consumed_total: int = 0
    ongoing_wait_s: float = 0.0
    window_cycles: int | None = None

    def validate(self):
        if not isinstance(self.run_id, str) or not self.run_id or len(self.run_id) > 128:
            raise ValueError("bounded run_id required")
        if self.unit not in ("prompt_group", "workflow_group"):
            raise ValueError("native sampling unit required")
        if self.phase not in ("training", "sync", "eval", "paused", "shutdown", "unknown"):
            raise ValueError("invalid pipeline phase")
        if self.window_cycles is not None and (
            type(self.window_cycles) is not int
            or not 0 <= self.window_cycles <= self.completed_cycles
        ):
            raise ValueError("invalid complete cycles in current window")
        for key in (
            "sequence",
            "topology_revision",
            "completed_cycles",
            "policy_version",
            "eligible_units",
            "next_batch_units",
            "consumed_total",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in (
            "available_work",
            "dispatch_unconstrained",
            "attribution_known",
            "eligibility_known",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("oldest_buffer_policy_version", "oldest_consumed_policy_version"):
            value = getattr(self, key)
            if value is not None and (
                type(value) is not int or value < 0 or value > self.policy_version
            ):
                raise ValueError(f"invalid {key}")
        for key in (
            "observed_at",
            "window_s",
            "wait_data_s",
            "wait_staleness_s",
            "train_path_s",
            "wait_unknown_s",
            "ongoing_wait_s",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.window_s <= 0 or self.elapsed > self.window_s + 1e-6:
            raise ValueError("window must cover disjoint critical-path timings")
        return self

    @property
    def elapsed(self):
        return self.wait_data_s + self.wait_staleness_s + self.train_path_s + self.wait_unknown_s

    @property
    def data_fraction(self):
        return self.wait_data_s / self.elapsed if self.elapsed else 0.0

    @property
    def total_fraction(self):
        return (
            (self.wait_data_s + self.wait_staleness_s + self.wait_unknown_s) / self.elapsed
            if self.elapsed
            else 1.0
        )

    def usable(self, now, ttl, revision):
        return (
            0 <= now - self.observed_at <= ttl
            and self.topology_revision == revision
            and self.phase == "training"
            and self.attribution_known
            and self.wait_unknown_s == 0
            and self.elapsed > 0
        )


def snapshot_from_body(body):
    return PipelineSnapshot(**body).validate()


class PipelineWindow:
    """Adapter-side duration accumulator; publish only outside sampling/NPU locks.

    Caller supplies native critical-path measurements, never per-rank summed
    timings. Buffer and version metadata must come from the framework snapshot.
    """

    def __init__(self, run_id, unit, max_cycles=10):
        from collections import deque

        if type(max_cycles) is not int or max_cycles < 3:
            raise ValueError("max_cycles must be at least three")
        self.run_id, self.unit = run_id, unit
        self.cycles = deque(maxlen=max_cycles)
        self.sequence = self.completed_cycles = self.consumed_total = 0
        self.revision = None
        self.lock = RLock()

    def record(
        self, *, wait_data_s, wait_staleness_s, train_path_s, consumed_units, wait_unknown_s=0
    ):
        with self.lock:
            return self._record(
                wait_data_s, wait_staleness_s, train_path_s, consumed_units, wait_unknown_s
            )

    def _record(self, wait_data_s, wait_staleness_s, train_path_s, consumed_units, wait_unknown_s):
        values = (wait_data_s, wait_staleness_s, train_path_s, wait_unknown_s)
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in values
        ):
            raise ValueError("invalid cycle timings")
        if type(consumed_units) is not int or consumed_units < 0:
            raise ValueError("invalid native consumed count")
        self.cycles.append(values)
        self.completed_cycles += 1
        self.consumed_total += consumed_units

    def snapshot(self, *, revision, observed_at, policy_version, window_s, **metadata):
        with self.lock:
            return self._snapshot(revision, observed_at, policy_version, window_s, metadata)

    def _snapshot(self, revision, observed_at, policy_version, window_s, metadata):
        if self.revision is not None and revision != self.revision:
            self.cycles.clear()
        self.revision = revision
        self.sequence += 1
        timings = [sum(c[i] for c in self.cycles) for i in range(4)]
        return asdict(
            PipelineSnapshot(
                run_id=self.run_id,
                sequence=self.sequence,
                topology_revision=revision,
                observed_at=observed_at,
                window_s=window_s,
                completed_cycles=self.completed_cycles,
                policy_version=policy_version,
                unit=self.unit,
                wait_data_s=timings[0],
                wait_staleness_s=timings[1],
                train_path_s=timings[2],
                wait_unknown_s=timings[3],
                consumed_total=self.consumed_total,
                **metadata,
            ).validate()
        )
