"""One single-node placement group per rollout process actor."""

from .ray import RayProcessLauncher, _ProcessActor


def validate_label_selector(selector):
    """The rollout pool uses exact node-label matches."""
    if not isinstance(selector, dict) or not selector or any(
        not isinstance(k, str)
        or not k
        or not isinstance(v, str)
        or v.startswith(("!", "in("))
        for k, v in selector.items()
    ):
        raise ValueError("ray_pg requires nonempty pg_label_selector with exact string labels")


class RayPGProcessLauncher(RayProcessLauncher):
    def __init__(self, *args, label_selector, **kwargs):
        super().__init__(*args, **kwargs)
        validate_label_selector(label_selector)
        self.label_selector = dict(label_selector)
        self.groups = {}
        self.pending = {}
        self.locations = {}

    def start(self, replica_id, lease, plan):
        from ray.util.placement_group import placement_group

        if replica_id in self.groups:
            return
        slot = lease.slot
        bundle = {
            "CPU": slot.num_cpus,
            "GPU" if plan.accelerator == "gpu" else "NPU": slot.num_devices,
        }
        pg = placement_group(
            [bundle], strategy="STRICT_PACK", bundle_label_selector=[self.label_selector]
        )
        self.groups[replica_id] = pg
        self.leases[replica_id] = lease
        self.pending[replica_id] = (pg.ready(), plan)

    def started(self, replica_id):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if replica_id in self.pending:
            ref, plan = self.pending[replica_id]
            ready, _ = self.ray.wait([ref], timeout=0)
            if not ready:
                return False
            self.ray.get(ref)
            slot = self.leases[replica_id].slot
            actor = (
                self.ray.remote(_ProcessActor)
                .options(
                    num_cpus=slot.num_cpus,
                    **(
                        {"num_gpus": slot.num_devices}
                        if plan.accelerator == "gpu"
                        else {"resources": {"NPU": slot.num_devices}}
                    ),
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.groups[replica_id],
                        placement_group_bundle_index=0,
                        placement_group_capture_child_tasks=False,
                    ),
                    max_restarts=0,
                    max_task_retries=0,
                )
                .remote()
            )
            self.actors[replica_id] = actor
            self.starts[replica_id] = actor.start.remote(
                plan.argv,
                slot.num_devices,
                slot.port,
                dict(plan.env),
                accelerator=plan.accelerator,
            )
            self.locations[replica_id] = actor.location.remote(slot.port)
            del self.pending[replica_id]
        if not super().started(replica_id):
            return False
        ref = self.locations[replica_id]
        ready, _ = self.ray.wait([ref], timeout=0)
        if not ready:
            return False
        self.leases[replica_id].allocation.endpoint = self.ray.get(ref)
        return True

    def stop(self, replica_id):
        from ray.util.placement_group import placement_group_table, remove_placement_group

        # Never remove an allocated PG until the process supervisor confirms cleanup.
        super().stop(replica_id)
        pg = self.groups.get(replica_id)
        if pg is None:
            return
        remove_placement_group(pg)
        if placement_group_table(pg).get("state") != "REMOVED":
            raise RuntimeError("placement group removal pending; lease retained")
        self.groups.pop(replica_id, None)
        self.pending.pop(replica_id, None)
        self.locations.pop(replica_id, None)
        self.leases.pop(replica_id, None)
