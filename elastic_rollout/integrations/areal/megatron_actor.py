"""Dense BF16 Megatron producer using AReaL's native Bridge export contract."""

import torch
from areal.engine.megatron_engine import MegatronPPOActor

from elastic_rollout.export import coordinated_export

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
        def native_export():
            yield from self.bridge.export_hf_weights(self.model, cpu=False, show_progress=False)

        return coordinated_export(
            native_export(),
            self._synchronize_bucket,
            transform=lambda tensor: tensor.to(dtype=torch.bfloat16).contiguous(),
            include_dtype=False,
        )
