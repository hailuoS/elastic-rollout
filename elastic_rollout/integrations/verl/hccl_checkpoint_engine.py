"""Verl CheckpointEngine facade over the shared Direct HCCL sender."""

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from elastic_rollout.hccl_sender import HCCLSender, weight_groups  # noqa: F401


@CheckpointEngineRegistry.register("elastic_hccl")
class ElasticHCCLCheckpointEngine(HCCLSender, CheckpointEngine):
    def _node_ip(self):
        from ray.util import get_node_ip_address

        return get_node_ip_address()
