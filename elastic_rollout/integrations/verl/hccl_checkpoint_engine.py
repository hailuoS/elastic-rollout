"""Verl CheckpointEngine facade over the shared Direct HCCL sender."""

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from elastic_rollout.export import coordinated_export, synchronize_manifest
from elastic_rollout.hccl_sender import HCCLSender, weight_groups  # noqa: F401


@CheckpointEngineRegistry.register("elastic_hccl")
class ElasticHCCLCheckpointEngine(HCCLSender, CheckpointEngine):
    def __init__(self, *args, synchronize_per_tensor=False, **kwargs):
        self._synchronize_per_tensor = synchronize_per_tensor
        super().__init__(*args, **kwargs)

    def prepare(self, transport_plan=None):
        if (
            transport_plan is not None
            or self.backend.name == "nccl"
            or self._synchronize_per_tensor
        ) and not hasattr(self, "_elastic_cpu_group"):
            from datetime import timedelta

            import torch.distributed as dist

            # Every training rank enters this native RPC once. Keep bucket/error
            # coordination off the inference HCCL groups being created/retired.
            self._elastic_cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        return super().prepare(transport_plan)

    def _synchronize_bucket(self, error, manifest):
        import torch.distributed as dist

        synchronize_manifest(dist, self._elastic_cpu_group, error, manifest)

    async def send_weights(self, weights, global_steps=None):
        if self._synchronize_per_tensor:
            weights = self._coordinated_export(weights)
        return self.send_weights_sync(
            weights,
            global_steps,
            synchronize_group=(
                self._synchronize_bucket if hasattr(self, "_elastic_cpu_group") else None
            ),
        )

    def _coordinated_export(self, weights):
        return coordinated_export(weights, self._synchronize_bucket)

    def _node_ip(self):
        from ray.util import get_node_ip_address

        return get_node_ip_address()
