"""CUDA facade over the same bounded sender and Core policy transaction."""

from verl.checkpoint_engine.base import CheckpointEngineRegistry

from .hccl_checkpoint_engine import ElasticHCCLCheckpointEngine


@CheckpointEngineRegistry.register("elastic_nccl")
class ElasticNCCLCheckpointEngine(ElasticHCCLCheckpointEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, backend="nccl", **kwargs)
