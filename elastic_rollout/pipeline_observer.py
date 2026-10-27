"""Local training observations and bounded, off-path control-plane reporting.

One observer belongs to the trainer controller, never to individual training
ranks. Counts are native groups from the global sampler/dispatcher; timings
are the controller's batch-fetch and training critical path, not rank sums.
"""

import logging
import time
from collections import deque
from contextlib import contextmanager
from functools import wraps
from threading import Event, RLock, Thread
from uuid import uuid4

from .pipeline import PipelineWindow

logger = logging.getLogger(__name__)


class PipelineObserver:
    def __init__(self, control, unit, metadata, *, interval=5.0, clock=time.monotonic):
        if interval <= 0:
            raise ValueError("report interval must be positive")
        self.control, self.metadata = control, metadata
        self.interval, self.clock = interval, clock
        self.window = PipelineWindow(uuid4().hex, unit)
        self.lock = RLock()
        self.phase, self.version = "paused", 0
        self.active = False
        self.acquiring = False
        self.segment = None
        self.segment_at = self.clock()
        self.timings = dict(data=0.0, staleness=0.0, train=0.0, unknown=0.0)
        self.consumed = 0
        self.consumed_version = None
        self.consumed_known = True
        self.versions = deque(maxlen=10)
        self.revision = None
        self.stop = Event()
        self.thread = None
        self.last_error = None
        self.last_body = None

    def start(self):
        if self.thread is not None:
            return
        self.thread = Thread(target=self._run, name="elastic-pipeline-reporter", daemon=True)
        self.thread.start()

    def _segment(self, kind):
        now = self.clock()
        if self.active and self.segment is not None and self.phase == "training":
            self.timings[self.segment] += max(0.0, now - self.segment_at)
        self.segment, self.segment_at = kind, now

    def begin(self):
        with self.lock:
            if self.active:
                raise RuntimeError("overlapping training consumption cycles")
            self.active, self.acquiring, self.phase = True, True, "training"
            self.timings = dict.fromkeys(self.timings, 0.0)
            self.consumed, self.consumed_version = 0, None
            self.consumed_known = True
            self.segment, self.segment_at = "train", self.clock()

    def acquired(self):
        with self.lock:
            self.acquiring = False

    @contextmanager
    def waiting(self, reason):
        with self.lock:
            self._segment(reason)
        try:
            yield
        finally:
            with self.lock:
                self._segment("train")

    def consumed_groups(self, count, oldest_version):
        with self.lock:
            self.consumed += count
            if count and oldest_version is None:
                self.consumed_known = False
            if oldest_version is not None:
                self.consumed_version = (
                    oldest_version
                    if self.consumed_version is None
                    else min(oldest_version, self.consumed_version)
                )

    def finish(self, success=True):
        with self.lock:
            if not self.active:
                return
            self._segment(None)
            if success:
                self.window.record(
                    wait_data_s=self.timings["data"],
                    wait_staleness_s=self.timings["staleness"],
                    wait_unknown_s=self.timings["unknown"],
                    train_path_s=self.timings["train"],
                    consumed_units=self.consumed,
                )
                self.versions.append(self.consumed_version if self.consumed_known else None)
            self.active = self.acquiring = False
            self.phase = "training" if success else "unknown"

    @contextmanager
    def excluded(self, phase):
        with self.lock:
            previous = self.phase
            self._segment(self.segment)
            self.phase = phase
        try:
            yield
        finally:
            with self.lock:
                self.phase, self.segment_at = previous, self.clock()

    def report_once(self):
        # No framework, sampler, NPU or observer lock is held across HTTP I/O.
        status = self.control.status()
        if not status["spec"].get("elastic_scaling"):
            return
        metadata = self.metadata(status)
        with self.lock:
            revision = status["scaling"]["revision"]
            if self.revision is not None and revision != self.revision:
                self.window.cycles.clear()
                self.versions.clear()
                # A cycle spanning a topology change cannot prove its effect.
                if self.active:
                    self.timings = dict.fromkeys(self.timings, 0.0)
                    self.timings["unknown"] = 1e-9
                    self.segment_at = self.clock()
            self.revision = revision
            elapsed = sum(sum(c) for c in self.window.cycles)
            versions = list(self.versions)
            body = self.window.snapshot(
                revision=revision,
                observed_at=time.time(),
                policy_version=self.version,
                window_s=max(elapsed, 1e-6),
                window_cycles=len(self.window.cycles),
                phase=self.phase,
                oldest_consumed_policy_version=(
                    min(versions) if versions and all(v is not None for v in versions) else None
                ),
                ongoing_wait_s=(
                    sum(self.timings[k] for k in ("data", "staleness", "unknown"))
                    + (max(0.0, self.clock() - self.segment_at) if self.segment != "train" else 0.0)
                    if self.active and self.acquiring and self.phase == "training"
                    else 0.0
                ),
                **metadata,
            )
            # Include unfinished starvation in expansion evidence, never in
            # completed-cycle counters or the evidence for low-wait scale-in.
            if self.active and self.phase == "training":
                partial = dict(self.timings)
                if self.segment is not None:
                    partial[self.segment] += max(0.0, self.clock() - self.segment_at)
                for key, field in (
                    ("data", "wait_data_s"),
                    ("staleness", "wait_staleness_s"),
                    ("unknown", "wait_unknown_s"),
                ):
                    body[field] += partial[key]
                body["window_s"] += sum(partial[k] for k in ("data", "staleness", "unknown"))
        self.control.publish_pipeline(body)
        self.last_error = None
        self.last_body = body
        return body

    def _run(self):
        while not self.stop.is_set():
            try:
                self.report_once()
            except Exception as exc:
                # Failure means absent evidence. Core's TTL cancels a shrink
                # trial; reporting must never stop training or fabricate zeros.
                error = f"{type(exc).__name__}: {exc}"
                if error != self.last_error:
                    logger.warning("pipeline report unavailable: %s", error)
                self.last_error = error
            self.stop.wait(self.interval)

    def close(self):
        with self.lock:
            self.phase = "shutdown"
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.last_body is not None and (self.thread is None or not self.thread.is_alive()):
            # Report shutdown without consulting a sampler that may already be
            # tearing down. If Core advanced meanwhile, its revision/version
            # guards reject this and the normal TTL remains the fallback.
            body = dict(
                self.last_body,
                sequence=self.window.sequence + 1,
                observed_at=time.time(),
                phase="shutdown",
            )
            try:
                self.control.publish_pipeline(body)
            except Exception:
                logger.debug("final pipeline shutdown report rejected", exc_info=True)


def wrap_method(instance, name, around):
    """Wrap one integration-owned instance, preserving its native bound method."""
    original = getattr(instance, name)

    @wraps(original)
    def wrapped(*args, **kwargs):
        return around(original, *args, **kwargs)

    setattr(instance, name, wrapped)


def exclude_method(instance, name, observer, phase):
    def around(original, *args, **kwargs):
        with observer.excluded(phase):
            return original(*args, **kwargs)

    wrap_method(instance, name, around)
