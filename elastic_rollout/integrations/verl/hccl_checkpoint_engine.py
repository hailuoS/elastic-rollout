"""Verl CheckpointEngine facade over the shared Direct HCCL sender."""

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

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
        # Bridge returns a global HF stream on every TP/PP/DP rank. Compare
        # each item before entering the next native export collective, rather
        # than discovering mismatched streams at different bucket boundaries.
        weights = iter(weights)
        try:
            while True:
                item = None
                error = None
                try:
                    name, tensor = next(weights)
                    item = (name, tensor)
                except StopIteration:
                    pass
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                self._synchronize_bucket(
                    error,
                    None
                    if item is None
                    else {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
                )
                if item is None:
                    return
                yield item
        finally:
            close = getattr(weights, "close", None)
            if close is not None:
                close()

    def _node_ip(self):
        from ray.util import get_node_ip_address

        return get_node_ip_address()
