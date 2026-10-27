"""CUDA facade over the same bounded sender and Core policy transaction."""

from datetime import timedelta

from verl.checkpoint_engine.base import CheckpointEngineRegistry

from .hccl_checkpoint_engine import ElasticHCCLCheckpointEngine


@CheckpointEngineRegistry.register("elastic_nccl")
class ElasticNCCLCheckpointEngine(ElasticHCCLCheckpointEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, backend="nccl", **kwargs)
        self.cpu_group = None

    def prepare(self):
        import torch.distributed as dist

        if self.cpu_group is None:
            # All FSDP ranks call prepare. A sender failure must be exchanged
            # before peers enter the next lazy full_tensor collective.
            self.cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        return super().prepare()

    def _synchronize_bucket(self, error, manifest):
        import torch.distributed as dist

        values = [None] * dist.get_world_size(group=self.cpu_group)
        dist.all_gather_object(values, {"error": error, "manifest": manifest}, group=self.cpu_group)
        if any(v["error"] for v in values):
            raise RuntimeError(f"distributed export failed: {[v['error'] for v in values]}")
        if any(v["manifest"] != values[0]["manifest"] for v in values):
            raise RuntimeError("FSDP ranks produced different weight manifests")

    async def send_weights(self, weights, global_steps=None):
        return self.send_weights_sync(
            weights, global_steps, synchronize_group=self._synchronize_bucket
        )
