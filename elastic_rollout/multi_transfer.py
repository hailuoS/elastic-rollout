"""Immutable base and per-replica HCCL transports; no tensor data enters Core."""

import threading
from copy import deepcopy
from uuid import uuid4

from .api import PolicyUpdateState as U
from .api import ReplicaState as S
from .runtime import PolicyConflict
from .weight_transfer import TransferCoordinator


class MultiTransferCoordinator:
    def __init__(self, manager):
        self.manager = manager
        self.children = {}
        self.entries = []
        self.completed_entries = {}
        self.optional_failures = {}
        self._session = None
        self.retirements = {}
        self.retired = set()
        self.receiver_destroying = set()
        self.receiver_destroyed = set()
        self.poisoned = False
        self.operation_lock = threading.Lock()

    @property
    def transport(self):
        child = self.children.get("base")
        return child.transport if child else None

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self._session = value
        self.entries = []
        self.completed_entries.clear()
        self.optional_failures.clear()
        for child in self.children.values():
            child.session = None

    def status(self):
        if self._session is None:
            return None
        with self.manager.traffic.lock:
            entries = []
            for plan in self.entries:
                key = plan["transport_key"]
                child = self.children.get(key)
                session = self._describe(child) if child else deepcopy(self.completed_entries[key])
                entries.append({**deepcopy(plan), "session": session})
            return {
                **self._session,
                "state": self.manager.policy_update.state.value,
                "transports": entries,
                "optional_failures": dict(self.optional_failures),
            }

    @staticmethod
    def _describe(child):
        t = child.transport
        return child.status() | {
            "master_address": t.master_address,
            "master_port": t.master_port,
            "world_size": t.world_size,
            "sender_instance_id": t.sender_instance_id,
        }

    def transport_status(self):
        return {
            "state": "POISONED" if self.poisoned else "READY",
            "transports": {key: child.transport_status() for key, child in self.children.items()},
        }

    def fail(self, error):
        self.poisoned = True
        for child in self.children.values():
            child.fail(error)

    def maintenance(self, body=None):
        m = self.manager
        if body and "destroy_receivers" in body:
            request = body["destroy_receivers"]
            with m.traffic.lock:
                key = request["transport_key"]
                if m.transaction_owned or request != self.retirements.get(key):
                    raise PolicyConflict("retirement identity or lifecycle changed")
                child = self.children[key]
            return self._destroy_receivers(child, request["retirement_id"], request["deadline_at"])
        with m.traffic.lock:
            if body:
                if body.get("error"):
                    self.poisoned = True
                    raise PolicyConflict("sender retirement failed; clean restart required")
                if m.transaction_owned:
                    raise PolicyConflict("retirement acknowledgement during policy transaction")
                for ack in body.get("acknowledgements", []):
                    key, token = ack["transport_key"], ack["retirement_id"]
                    if token in self.retired:
                        continue
                    request = self.retirements[key]
                    if request["retirement_id"] != token:
                        raise PolicyConflict("retirement identity mismatch")
                    if token not in self.receiver_destroyed:
                        raise PolicyConflict("receiver destroy acknowledgement required")
                    if request["deadline_at"] <= m.clock():
                        self.poisoned = True
                        raise PolicyConflict("retirement deadline expired; clean restart required")
                    self.retired.add(token)
                    del self.retirements[key]
                    if any(entry["transport_key"] == key for entry in self.entries):
                        # Retired sessions are immutable history, retained only
                        # until the next BEGIN. Live sessions are never cached.
                        self.completed_entries[key] = self._describe(self.children[key])
                    del self.children[key]
                    m._transition(m.traffic.replicas[key], S.STOPPING)
            for request in self.retirements.values():
                if request["deadline_at"] <= m.clock():
                    self.poisoned = True
            return {"requests": list(self.retirements.values()), "poisoned": self.poisoned}

    def _destroy_receivers(self, child, token, deadline):
        m = self.manager
        with m.traffic.lock:
            if token in self.receiver_destroyed:
                return {"receiver_destroyed": token}
            if token in self.receiver_destroying or deadline <= m.clock():
                raise PolicyConflict("receiver retirement in progress or expired")
            self.receiver_destroying.add(token)
        try:
            child._fanout(
                child.transport.targets,
                lambda target: m.driver.weight_transfer_control.destroy_weight_transfer(
                    target.endpoint,
                    m.spec.replica.weight_sync,
                    token,
                    target.worker_world_size,
                    timeout=min(120, deadline - m.clock()),
                ),
            )
        except BaseException:
            self.poisoned = True
            raise
        with m.traffic.lock:
            self.receiver_destroyed.add(token)
        return {"receiver_destroyed": token}

    def request_retirement(self, replica):
        with self.manager.traffic.lock:
            return self._request_retirement(replica)

    def _request_retirement(self, replica):
        key = replica.replica_id
        child = self.children.get(key)
        if child is None or child.transport is None:
            return False
        self.retirements.setdefault(
            key,
            {
                "transport_key": key,
                "transport_id": child.transport.transport_id,
                "retirement_id": uuid4().hex,
                "deadline_at": self.manager.clock() + self.manager.spec.retirement_timeout_s,
            },
        )
        self.manager._transition(replica, S.RETIRE_PENDING)
        return True

    def execute(self, update_id, action, body):
        m = self.manager
        with m.traffic.lock:
            if (
                not m.policy_update
                or m.policy_update.update_id != update_id
                or m.policy_update.state in (U.FAILED, U.COMMITTED)
                or m.closing
            ):
                raise PolicyConflict("no active external update")
        if action == "fail":
            m._fail_update(RuntimeError(body.get("error", "sender failed")))
            return m.get_policy_update(update_id)
        if not self.operation_lock.acquire(blocking=False):
            raise PolicyConflict("transfer operation already in progress")
        try:
            return self._execute(update_id, action, body)
        finally:
            self.operation_lock.release()

    def _execute(self, update_id, action, body):
        m = self.manager
        if self.poisoned:
            raise PolicyConflict("multi-transport poisoned; clean restart required")
        if action in ("destroy", "isolate"):
            key = body.get("transport_key")
            plan = next((p for p in self.entries if p["transport_key"] == key), None)
            if not plan or not plan["optional"]:
                raise PolicyConflict(
                    "only an optional transport with sender destroy proof may isolate"
                )
            if key not in self.children:
                return {"isolated": key}
            child = self.children[key]
            if (
                body.get("session_id") != child.session.session_id
                or body.get("transport_id") != child.transport.transport_id
                or child.operation_lock.locked()
            ):
                raise PolicyConflict(
                    "optional transport identity changed or operation still running"
                )
            token = child.session.session_id
            if action == "destroy":
                return self._destroy_receivers(child, token, m.policy_update.deadline_at)
            if body.get("sender_destroyed") is not True or token not in self.receiver_destroyed:
                raise PolicyConflict("sender and receiver destroy acknowledgements required")
            with m.traffic.lock:
                error = str(body.get("error", "optional transfer failed"))
                self.optional_failures[key] = error
                child.fail(error)
                replica = m.traffic.replicas[key]
                replica.cancel_requested = True
                replica.error = error
                replica.prepared_update_id = replica.prepared_policy_version = None
                m.scale_failures += 1
                self.completed_entries[key] = self._describe(child)
                del self.children[key]
            return {"isolated": key}
        if action == "prepare":
            if self._session is not None:
                raise PolicyConflict("transfer session already exists")
            sources = body["transports"]
            addresses = [
                (
                    sources[p["transport_key"]]["master_address"],
                    sources[p["transport_key"]]["master_port"],
                )
                for p in m.policy_update.transport_plan
            ]
            if len(set(addresses)) != len(addresses):
                raise ValueError("each HCCL transport needs a distinct rendezvous")
            m._update_transition(U.PREPARING)
            for plan in m.policy_update.transport_plan:
                key = plan["transport_key"]
                child = self.children.setdefault(
                    key,
                    TransferCoordinator(
                        m,
                        tuple(plan["replica_ids"]),
                        optional=plan["optional"],
                        cohort=True,
                    ),
                )
                # A successful join is required on every subsequent transaction.
                child.optional = plan["optional"]
                try:
                    child.execute(
                        update_id, action, {"backend": m.spec.replica.weight_sync, **sources[key]}
                    )
                except Exception as exc:
                    if not plan["optional"] or (child.session and child.session.physical_started):
                        m._fail_update(exc)
                        raise
                    replica = m.traffic.replicas[key]
                    replica.cancel_requested = True
                    m.scale_failures += 1
                    replica.error = f"optional discovery failed: {exc}"
                    continue
                self.entries.append(dict(plan))
            self._session = {
                "session_id": uuid4().hex,
                "update_id": update_id,
                "target_version": m.policy_update.target_version,
                "needs_initialize": any(
                    self.children[e["transport_key"]].session.needs_initialize for e in self.entries
                ),
            }
            return self.status()
        if body.get("transport_key"):
            key = body["transport_key"]
            if key not in {e["transport_key"] for e in self.entries}:
                raise PolicyConflict("transport is not frozen in this transaction")
            try:
                return self.children[key].execute(update_id, action, body)
            except Exception as exc:
                if next(e for e in self.entries if e["transport_key"] == key)["optional"]:
                    # A returned HTTP operation alone is not isolation proof.
                    # Sender must synchronize and destroy its communicator, then
                    # explicitly acknowledge isolate before any policy COMMIT.
                    self.optional_failures[key] = str(exc)
                    return {"optional_failure": str(exc), "transport_key": key}
                m._fail_update(exc)
                raise
        if not self._session or body.get("session_id") != self._session["session_id"]:
            raise PolicyConflict("multi-transfer session identity mismatch")
        if action not in ("start", "finish"):
            raise ValueError("transport_key required for collective operation")
        try:
            counts = []
            for entry in self.entries:
                key = entry["transport_key"]
                if key not in self.children:
                    continue  # Acknowledged optional sender destroy.
                if key in self.optional_failures:
                    continue  # Cannot send more; settle_optional must isolate before COMMIT.
                child = self.children[key]
                try:
                    result = child.execute(
                        update_id, action, {"session_id": child.session.session_id}
                    )
                except Exception as exc:
                    if not entry["optional"]:
                        raise
                    self.optional_failures[key] = str(exc)
                    continue
                counts.append(result["num_weight_groups"])
            if action == "finish" and len(set(counts)) != 1:
                raise RuntimeError("transport weight group counts differ")
            m._update_transition(U.TRANSFERRING if action == "start" else U.PREPARED)
            return self.status()
        except Exception as exc:
            m._fail_update(exc)
            raise
