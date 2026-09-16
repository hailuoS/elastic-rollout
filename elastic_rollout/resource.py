"""Static resource inventory; Ray enforces the actual accelerator reservation."""

from threading import RLock
from uuid import uuid4

from .api import ResourceLease, ResourceSlot


class NoCapacity(RuntimeError):
    pass


class StaticResourceProvider:
    def __init__(self, slots: list[ResourceSlot]):
        self.slots = {s.slot_id: s for s in slots}
        if len(self.slots) != len(slots):
            raise ValueError("duplicate slot_id")
        for i, slot in enumerate(slots):
            for other in slots[:i]:
                if slot.node_id == other.node_id or slot.host == other.host:
                    if slot.port == other.port:
                        raise ValueError(
                            f"Slots {other.slot_id}/{slot.slot_id} on node {slot.node_id} "
                            f"(host {slot.host}) conflict on serving port {slot.port}"
                        )
        self._leases: dict[str, ResourceLease] = {}
        self._lock = RLock()

    def acquire(self, replica_id: str) -> ResourceLease:
        with self._lock:
            for lease in self._leases.values():
                if lease.replica_id == replica_id:
                    return lease
            for slot_id, slot in self.slots.items():
                if slot_id not in self._leases:
                    lease = ResourceLease(uuid4().hex, replica_id, slot)
                    self._leases[slot_id] = lease
                    return lease
            raise NoCapacity("no free static resource slot")

    def release(self, lease: ResourceLease) -> bool:
        with self._lock:
            if self._leases.get(lease.slot.slot_id) != lease:
                return False
            del self._leases[lease.slot.slot_id]
            return True

    @property
    def allocated(self) -> int:
        with self._lock:
            return len(self._leases)
