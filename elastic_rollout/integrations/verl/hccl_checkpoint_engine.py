"""Verl CheckpointEngine facade over the shared Direct HCCL sender."""

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from elastic_rollout.hccl_sender import HCCLSender, weight_groups  # noqa: F401


@CheckpointEngineRegistry.register("elastic_hccl")
class ElasticHCCLCheckpointEngine(HCCLSender, CheckpointEngine):
    def prepare(self, transport_plan=None):
        if transport_plan is not None and not hasattr(self, "_elastic_cpu_group"):
            from datetime import timedelta

            import torch.distributed as dist

            # Every FSDP rank enters this native RPC once. Keep bucket/error
            # coordination off the inference HCCL groups being created/retired.
            self._elastic_cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        return super().prepare(transport_plan)

    def _synchronize_bucket(self, error, manifest):
        import torch.distributed as dist

        values = [None] * dist.get_world_size(group=self._elastic_cpu_group)
        dist.all_gather_object(
            values, {"error": error, "manifest": manifest}, group=self._elastic_cpu_group
        )
        errors = [value["error"] for value in values if value["error"]]
        if errors:
            raise RuntimeError(f"distributed HCCL export failed: {errors}")
        if any(value["manifest"] != values[0]["manifest"] for value in values):
            raise RuntimeError("training ranks produced different HCCL weight manifests")

    async def send_weights(self, weights, global_steps=None):
        return self.send_weights_sync(
            weights,
            global_steps,
            synchronize_group=(
                self._synchronize_bucket if hasattr(self, "_elastic_cpu_group") else None
            ),
        )

    def _node_ip(self):
        from ray.util import get_node_ip_address

        return get_node_ip_address()
