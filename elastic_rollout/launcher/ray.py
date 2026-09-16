"""Ray actors reserve single-node NPU slots and own independent vLLM processes."""

import os
import socket
import subprocess
import sys

from elastic_rollout.api import ResourceLease

from .base import LaunchPlan


class _ProcessActor:
    def __init__(self):
        self.process = None

    def start(self, argv: tuple[str, ...], expected_devices: tuple[str, ...], port: int) -> None:
        if os.name != "posix":
            raise RuntimeError("RayProcessLauncher requires Linux Ascend worker nodes")
        # Never override Ray's device assignment: validate static inventory against
        # the actual Ray reservation, failing before launch if they disagree.
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if set(visible) != set(expected_devices):
            raise RuntimeError(f"Ray assigned NPU IDs {visible}, expected {expected_devices}")
        if self.process is not None:
            raise RuntimeError("process already started")
        # Refuse an already occupied API port before any health probe can
        # accidentally identify a pre-existing server as our new replica.
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))
        self.process = subprocess.Popen(
            [sys.executable, "-m", "elastic_rollout.launcher.supervisor", *argv],
            stdin=subprocess.PIPE,
        )

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait(timeout=20)


class RayProcessLauncher:
    def __init__(self, address: str = "auto", namespace: str = "elastic-rollout"):
        import ray

        if not ray.is_initialized():
            ray.init(address=address, namespace=namespace)
        self.ray = ray
        self.actors: dict = {}
        self.starts: dict = {}

    def start(self, replica_id: str, lease: ResourceLease, plan: LaunchPlan) -> None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        if replica_id in self.actors:
            return
        slot = lease.slot
        actor = (
            self.ray.remote(_ProcessActor)
            .options(
                num_cpus=slot.num_cpus,
                resources={"NPU": len(slot.device_ids)},
                scheduling_strategy=NodeAffinitySchedulingStrategy(slot.node_id, soft=False),
                max_restarts=0,
                max_task_retries=0,
            )
            .remote()
        )
        self.actors[replica_id] = actor
        self.starts[replica_id] = actor.start.remote(plan.argv, slot.device_ids, slot.port)

    def started(self, replica_id: str) -> bool:
        ready, _ = self.ray.wait([self.starts[replica_id]], timeout=0)
        if not ready:
            return False
        self.ray.get(ready[0])
        return True

    def alive(self, replica_id: str) -> bool:
        if not self.started(replica_id):
            return True  # Pending resource scheduling is bounded by startup_timeout.
        return self.ray.get(self.actors[replica_id].alive.remote(), timeout=5)

    def stop(self, replica_id: str) -> None:
        actor = self.actors.get(replica_id)
        if actor is None:
            return
        # Even a pending start can race with process creation. Always require a
        # stop acknowledgement before releasing its physical slot.
        try:
            self.ray.get(actor.stop.remote(), timeout=25)
        except self.ray.exceptions.RayActorError:
            # Cannot prove remote process cleanup during a node partition.
            raise RuntimeError("actor unreachable; physical slot remains quarantined") from None
        self.ray.kill(actor, no_restart=True)
        del self.actors[replica_id]
        del self.starts[replica_id]
