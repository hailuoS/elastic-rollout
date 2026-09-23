"""Metadata-only, fixed-membership Direct HCCL preparation (MVP-2A)."""

import json
import logging
import math
import threading
import time
from concurrent.futures import Future, as_completed
from dataclasses import asdict, dataclass, field
from typing import Protocol
from uuid import uuid4

from .api import PolicyUpdateState as U
from .api import ReplicaState as S

logger = logging.getLogger(__name__)
METADATA_LIMIT = 32768
RECOVERY_REQUIRED = (
    "persistent HCCL sender/topology changed or transport poisoned; "
    "restart the HCCL deployment/trainer from a clean bootstrap"
)


def submit_hccl_control(function, *args, **kwargs):
    """One daemon HTTP control call, without ThreadPoolExecutor's exit join.

    Running I/O is not cancellable and retains its existing timeout. After a
    fail-closed operation abandons it, it must not keep the process alive.
    Only control I/O runs here; trainer NPU calls stay on the actor thread.
    """
    future = Future()

    def run():
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    threading.Thread(target=run, name="elastic-hccl-control", daemon=True).start()
    return future


class WeightTransferControl(Protocol):
    def get_world_size(self, endpoint: str, *, timeout: float = 30) -> int: ...
    def init_weight_transfer(self, endpoint: str, init_info: dict, *, timeout: float = 120): ...
    def start_weight_update(self, endpoint: str, *, timeout: float = 120): ...
    def receive_weight_group(self, endpoint: str, update_info: dict, *, timeout: float = 120): ...
    def finish_weight_update(self, endpoint: str, *, timeout: float = 120): ...
    def is_paused(self, endpoint: str, *, timeout: float = 3) -> bool: ...


def validate_group(info):
    allowed = {
        "names",
        "dtype_names",
        "shapes",
        "packed",
        "packed_buffer_size_bytes",
        "packed_num_buffers",
    }
    if set(info) != allowed:
        raise ValueError("weight group accepts only tensor metadata and packing options")
    if len(json.dumps(info).encode()) > METADATA_LIMIT:
        raise ValueError("weight group metadata exceeds 32768 bytes")
    names, dtypes, shapes = info["names"], info["dtype_names"], info["shapes"]
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("weight names must be a nonempty unique list")
    if not (len(names) == len(dtypes) == len(shapes)):
        raise ValueError("weight metadata lengths differ")
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("weight names must be nonempty strings")
    if any(d != "bfloat16" for d in dtypes):
        raise ValueError("MVP-2A supports dense BF16 weights only")
    if any(not isinstance(s, list) or any(type(n) is not int or n <= 0 for n in s) for s in shapes):
        raise ValueError("invalid tensor shapes")
    if type(info["packed"]) is not bool:
        raise ValueError("packed must be boolean")
    for key in ("packed_buffer_size_bytes", "packed_num_buffers"):
        if type(info[key]) is not int or info[key] <= 0:
            raise ValueError(f"{key} must be positive")
    size = sum(math.prod(shape) * 2 for shape in shapes)
    if size > info["packed_buffer_size_bytes"]:
        raise ValueError("weight group exceeds configured data bucket")
    return size


@dataclass(frozen=True)
class WeightTransferTarget:
    replica_id: str
    endpoint: str
    worker_world_size: int
    rank_offset: int


@dataclass
class HCCLTransport:
    transport_id: str
    sender_instance_id: str
    master_address: str
    master_port: int
    world_size: int
    targets: tuple[WeightTransferTarget, ...]
    target_signature: tuple[tuple[str, float, int], ...]
    state: str = "INITIALIZING"
    error: str | None = None
    init_count: int = 0


@dataclass
class WeightTransferSession:
    session_id: str
    update_id: str
    target_version: int
    transport_id: str
    needs_initialize: bool
    state: str = "PLANNED"
    num_weight_groups: int = 0
    bytes_transferred: int = 0
    error: str | None = None
    timings: dict = field(default_factory=dict)
    physical_started: bool = False


class TransferCoordinator:
    def __init__(self, manager):
        self.manager = manager
        self.session = None
        self.transport = None
        self.operation_lock = threading.Lock()

    def status(self):
        return asdict(self.session) if self.session else None

    def transport_status(self):
        t = self.transport
        if t is None:
            return None
        return {
            "transport_id": t.transport_id,
            "state": t.state,
            "world_size": t.world_size,
            "target_count": len(t.targets),
            "init_count": t.init_count,
            "error": t.error,
        }

    def _poison(self, error):
        with self.manager.traffic.lock:
            if self.transport:
                self.transport.state, self.transport.error = "POISONED", str(error)

    def fail(self, error):
        if self.session and self.session.physical_started:
            self._poison(error)
        if self.session and self.session.state != "FAILED":
            self.session.state, self.session.error = "FAILED", error

    def _check(self, update_id):
        from .runtime import PolicyConflict

        m = self.manager
        u = m.policy_update
        if (
            not m._external_update()
            or u.update_id != update_id
            or u.state in (U.FAILED, U.COMMITTED)
            or m.closing
        ):
            raise PolicyConflict("no active external update")
        if m._check_update_deadline():
            raise PolicyConflict("external update deadline exceeded")
        if not m._generation_pause_verified or sum(m.traffic.inflight.values()):
            raise PolicyConflict("verified generation pause and inflight=0 required")
        targets = [m.traffic.replicas[rid] for rid in u.target_replica_ids]
        if any(r.state != S.READY for r in targets):
            raise RuntimeError("frozen transfer target is no longer READY")
        if self.transport and self.transport.state == "POISONED":
            raise PolicyConflict(RECOVERY_REQUIRED)
        if self.session and self.session.update_id == update_id and self.transport:
            expected = self.transport.target_signature
            current = tuple(
                (r.replica_id, r.started_at, t.worker_world_size)
                for r, t in zip(sorted(targets, key=lambda r: r.replica_id), self.transport.targets)
            )
            if current != expected:
                self._poison(RECOVERY_REQUIRED)
                raise PolicyConflict(RECOVERY_REQUIRED)
        return targets

    def _fanout(self, targets, function):
        # Never hold the routing/lifecycle lock in network I/O. Every receiver
        # must enter the collective before any single response can complete.
        futures = [submit_hccl_control(function, target) for target in targets]
        try:
            return [f.result() for f in as_completed(futures)]
        finally:
            # A failed receiver must fail the session promptly. Other HTTP calls
            # have bounded deadlines and cannot advance state when they return.
            for future in futures:
                future.cancel()  # Running daemon calls are abandoned, not joined.

    def execute(self, update_id, action, body):
        from .runtime import PolicyConflict

        m = self.manager
        if action == "fail":
            with m.traffic.lock:
                if not m.policy_update or m.policy_update.update_id != update_id:
                    raise PolicyConflict("not the current update")
                m._fail_update(RuntimeError(str(body.get("error", "sender failed"))))
                return m.get_policy_update(update_id)
        if not self.operation_lock.acquire(blocking=False):
            raise PolicyConflict("transfer operation already in progress")
        started = time.monotonic()
        executing = False
        try:
            with m.traffic.lock:
                replicas = self._check(update_id)
                u = m.policy_update
                timeout = min(120, u.deadline_at - m.clock())
                if action == "prepare":
                    if self.session and self.session.update_id == update_id:
                        raise PolicyConflict("transfer session already exists")
                    if body.get("backend") != "hccl":
                        raise ValueError("backend must be hccl")
                    if (
                        not isinstance(body.get("master_address"), str)
                        or not body["master_address"]
                    ):
                        raise ValueError("master_address required")
                    port = body.get("master_port")
                    if type(port) is not int or not 1 <= port <= 65535:
                        raise ValueError("invalid master_port")
                    if (
                        not isinstance(body.get("sender_instance_id"), str)
                        or not body["sender_instance_id"]
                    ):
                        raise ValueError("stable sender_instance_id required")
                    m._update_transition(U.PREPARING)
                else:
                    if not self.session or body.get("session_id") != self.session.session_id:
                        raise PolicyConflict("transfer session identity mismatch")
                    expected = {
                        "initialize": "PLANNED",
                        "start": "INITIALIZED",
                        "group": "STARTED",
                        "finish": "STARTED",
                    }
                    if action not in expected or self.session.state != expected[action]:
                        raise PolicyConflict("invalid transfer operation/state")
                    if action == "group":
                        if body.get("group_index") != self.session.num_weight_groups:
                            raise PolicyConflict(
                                "weight group out of order; retries are not allowed"
                            )
                        byte_count = validate_group(body["update_info"])
                    if action == "finish" and not self.session.num_weight_groups:
                        raise PolicyConflict("cannot finish an empty transfer")
                    if action == "start":
                        # In-place mutation may leave mixed weights on failure.
                        # Do not advertise the old physical loaded version.
                        for replica in replicas:
                            replica.loaded_policy_version = None
                session = self.session
                if action in ("initialize", "start", "group", "finish"):
                    session.physical_started = True
                    if action == "initialize":
                        self.transport.init_count += 1
                executing = True

            def remaining():
                budget = started + timeout - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("transfer control operation deadline exceeded")
                return budget

            control = m.driver.weight_transfer_control
            if action == "prepare":

                def discover(r):
                    return r, control.get_world_size(r.endpoint, timeout=remaining())

                worlds = self._fanout(replicas, discover)
                offset = 1
                targets = []
                signature = []
                for r, world in sorted(worlds, key=lambda item: item[0].replica_id):
                    if world != m.spec.replica.tensor_parallel_size:
                        if self.transport:
                            self._poison(RECOVERY_REQUIRED)
                            raise PolicyConflict(RECOVERY_REQUIRED)
                        raise ValueError("receiver world size differs from allocated TP")
                    targets.append(WeightTransferTarget(r.replica_id, r.endpoint, world, offset))
                    signature.append((r.replica_id, r.started_at, world))
                    offset += world
            else:

                def call(target):
                    endpoint = target.endpoint
                    if action == "initialize":
                        control.init_weight_transfer(
                            endpoint,
                            {
                                "master_address": self.transport.master_address,
                                "master_port": self.transport.master_port,
                                "world_size": self.transport.world_size,
                                "rank_offset": target.rank_offset,
                            },
                            timeout=remaining(),
                        )
                    elif action == "start":
                        control.start_weight_update(endpoint, timeout=remaining())
                    elif action == "group":
                        control.receive_weight_group(
                            endpoint, body["update_info"], timeout=remaining()
                        )
                    else:
                        control.finish_weight_update(endpoint, timeout=remaining())
                        if not control.is_paused(endpoint, timeout=remaining()):
                            raise RuntimeError("receiver resumed before commit")

                self._fanout(self.transport.targets, call)
            with m.traffic.lock:
                self._check(update_id)  # Reject late success after deadline/crash/failure.
                if action == "prepare":
                    t = self.transport
                    if t is not None:
                        if (
                            t.state != "READY"
                            or t.sender_instance_id != body["sender_instance_id"]
                            or t.master_address != body["master_address"]
                            or t.master_port != port
                            or t.target_signature != tuple(signature)
                            or t.targets != tuple(targets)
                        ):
                            self._poison(RECOVERY_REQUIRED)
                            raise PolicyConflict(RECOVERY_REQUIRED)
                    else:
                        self.transport = HCCLTransport(
                            uuid4().hex,
                            body["sender_instance_id"],
                            body["master_address"],
                            port,
                            offset,
                            tuple(targets),
                            tuple(signature),
                        )
                    self.session = session = WeightTransferSession(
                        uuid4().hex,
                        update_id,
                        u.target_version,
                        self.transport.transport_id,
                        t is None,
                        state="PLANNED" if t is None else "INITIALIZED",
                    )
                elif action == "initialize":
                    session.state = "INITIALIZED"
                    self.transport.state = "READY"
                elif action == "start":
                    session.state = "STARTED"
                    m._update_transition(U.TRANSFERRING)
                elif action == "group":
                    session.num_weight_groups += 1
                    session.bytes_transferred += byte_count
                else:
                    m._update_transition(U.VERIFYING)
                    for r in replicas:
                        r.loaded_policy_version = session.target_version
                    session.state = "PREPARED"
                    m._update_transition(U.PREPARED)
                session.timings[action] = (
                    session.timings.get(action, 0) + time.monotonic() - started
                )
                logger.info(
                    "weight transfer update=%s session=%s action=%s target=%s state=%s",
                    update_id,
                    session.session_id,
                    action,
                    session.target_version,
                    session.state,
                )
                result = self.status()
                # Rendezvous belongs to the persistent transport. The sender
                # helper needs these scalars, never physical serving endpoints.
                result.update(
                    master_address=self.transport.master_address,
                    master_port=self.transport.master_port,
                    world_size=self.transport.world_size,
                    sender_instance_id=self.transport.sender_instance_id,
                )
                return result
        except Exception as exc:
            if executing or (self.transport and self.transport.state == "POISONED"):
                m._fail_update(exc)
                raise RuntimeError(f"transfer {action} failed: {exc}") from exc
            raise
        finally:
            self.operation_lock.release()
