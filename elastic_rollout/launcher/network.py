"""Network checks used during process cleanup."""

import logging
import socket
import time

logger = logging.getLogger(__name__)


def rollout_ports_closed(slots: list[dict], timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        open_slots = []
        for slot in slots:
            try:
                with socket.create_connection((slot["host"], slot["port"]), timeout=0.5):
                    open_slots.append(slot["slot_id"])
            except OSError:
                pass
        if not open_slots:
            return True
        if time.monotonic() >= deadline:
            logger.error("rollout serving ports remain open after Core close: %s", open_slots)
            return False
        time.sleep(0.2)
