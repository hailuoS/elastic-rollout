"""Dense BF16 Megatron producer using AReaL's native Bridge export contract."""

import torch
from areal.engine.megatron_engine import MegatronPPOActor

from .hccl_actor import ElasticHCCLActorMixin


class ElasticHCCLMegatronActor(ElasticHCCLActorMixin, MegatronPPOActor):
    def configure_elastic_hccl(self, options):
        if (
            self.bridge_cls != "megatron-bridge"
            or not self.mcore_config.use_bridge_for_update_weights
            or self.enable_fp8
            or self.lora_mode
            or self.parallel_strategy.expert_parallel_size != 1
            or self.parallel_strategy.context_parallel_size != 1
        ):
            raise ValueError(
                "external Megatron requires dense BF16 Bridge export without EP/CP/LoRA"
            )
        super().configure_elastic_hccl(options)

    def _elastic_weights(self, meta):
        # Same API as native _update_weights_via_bridge. All TP/PP/DP ranks
        # exhaust the global HF stream; Bridge owns gathering and conversion.
        def native_export():
            yield from self.bridge.export_hf_weights(self.model, cpu=False, show_progress=False)

        weights = native_export()
        try:
            while True:
                item = None
                error = None
                try:
                    name, tensor = next(weights)
                    item = (name, tensor.to(dtype=torch.bfloat16).contiguous())
                except StopIteration:
                    pass
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                self._synchronize_bucket(
                    error,
                    None if item is None else {"name": item[0], "shape": list(item[1].shape)},
                )
                if item is None:
                    return
                yield item
        finally:
            close = getattr(weights, "close", None)
            if close is not None:
                close()
