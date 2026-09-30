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
        self.actual_device_ids: tuple[str, ...] = ()

    def start(
        self, argv: tuple[str, ...], num_devices: int, port: int, env: dict | None = None
    ) -> tuple[str, ...]:
        import ray

        if os.name != "posix":
            raise RuntimeError("RayProcessLauncher requires Linux Ascend worker nodes")
        if self.process is not None:
            raise RuntimeError("process already started")
        # NOSET actors may see the whole node. Only Ray's runtime assignment owns
        # the device set; sort these local IDs without selecting any other NPUs.
        assigned = ray.get_runtime_context().get_accelerator_ids().get("NPU", [])
        if (
            len(assigned) != num_devices
            or any(not isinstance(x, str) or not x.isascii() or not x.isdecimal() for x in assigned)
            or len({int(x) for x in assigned}) != num_devices
        ):
            raise RuntimeError(
                f"Ray assigned invalid NPU IDs {assigned!r}; expected {num_devices} distinct "
                "numeric local device IDs"
            )
        self.actual_device_ids = tuple(sorted(assigned, key=int))
        child_env = os.environ.copy()
        child_env.update(env or {})
        # Launch capabilities cannot override Ray's physical device assignment.
        child_env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(self.actual_device_ids)
        # Refuse an already occupied API port before any health probe can
        # accidentally identify a pre-existing server as our new replica.
        with socket.socket() as probe:
            # Linux keeps closed server connections in TIME_WAIT across a
            # policy restart. Allow their reuse, while still rejecting a live
            # listener; SO_REUSEPORT must not be enabled here.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", port))
                probe.listen(1)
            except OSError as exc:
                raise OSError(
                    exc.errno, f"vLLM API port {port} unavailable on this node: {exc.strerror}"
                ) from exc
        self.process = subprocess.Popen(
            [sys.executable, "-m", "elastic_rollout.launcher.supervisor", *argv],
            stdin=subprocess.PIPE,
            env=child_env,
        )
        return self.actual_device_ids

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> tuple[str, ...]:
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait(timeout=20)
        return self.actual_device_ids


class RayProcessLauncher:
    def __init__(
        self,
        address: str = "auto",
        namespace: str = "elastic-rollout",
        expected_gcs: str | None = None,
    ):
        import ray

        if not ray.is_initialized():
            context = ray.init(address=address, namespace=namespace)
        else:
            context = ray.get_runtime_context()
        if expected_gcs is not None:
            info = getattr(context, "address_info", None)
            actual = info.get("gcs_address") if isinstance(info, dict) else None
            if actual != expected_gcs:
                raise RuntimeError(
                    f"Core connected to Ray GCS {actual!r}, expected {expected_gcs!r}"
                )
        self.ray = ray
        self.actors: dict = {}
        self.starts: dict = {}
        self.leases: dict[str, ResourceLease] = {}

    def start(self, replica_id: str, lease: ResourceLease, plan: LaunchPlan) -> None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        if replica_id in self.actors:
            return
        slot = lease.slot
        actor = (
            self.ray.remote(_ProcessActor)
            .options(
                num_cpus=slot.num_cpus,
                resources={"NPU": slot.num_devices},
                scheduling_strategy=NodeAffinitySchedulingStrategy(slot.node_id, soft=False),
                max_restarts=0,
                max_task_retries=0,
            )
            .remote()
        )
        self.actors[replica_id] = actor
        self.leases[replica_id] = lease
        self.starts[replica_id] = actor.start.remote(
            plan.argv, slot.num_devices, slot.port, dict(plan.env)
        )

    def started(self, replica_id: str) -> bool:
        ready, _ = self.ray.wait([self.starts[replica_id]], timeout=0)
        if not ready:
            return False
        self.leases[replica_id].allocation.actual_device_ids = tuple(self.ray.get(ready[0]))
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
            # Also recover assignment when start raised (e.g. an occupied port),
            # or a startup timeout occurred before we consumed the start result.
            assigned = self.ray.get(actor.stop.remote(), timeout=25)
            self.leases[replica_id].allocation.actual_device_ids = tuple(assigned)
        except self.ray.exceptions.RayActorError:
            # Cannot prove remote process cleanup during a node partition.
            raise RuntimeError("actor unreachable; physical slot remains quarantined") from None
        self.ray.kill(actor, no_restart=True)
        del self.actors[replica_id]
        del self.starts[replica_id]
        del self.leases[replica_id]
