"""Verl distributed producer adapter for the shared policy-update transaction."""

import asyncio
import os

import ray
from verl.utils.ray_utils import auto_await

from elastic_rollout.client import ControlClient
from elastic_rollout.weight_sync import ElasticWeightSync


class ElasticCheckpointEngineManager(ElasticWeightSync):
    def __init__(self, config, actor_wg, replicas):
        if config.backend not in ("elastic_hccl", "elastic_nccl") or replicas:
            raise ValueError("elastic_hccl requires external service get_replicas() == []")
        self.engine_backend = config.backend
        options = config.engine_kwargs[config.backend]
        self.control = ControlClient(
            options["endpoint"], options["deployment_id"], os.getenv("ELASTIC_ROLLOUT_TOKEN")
        )
        self.bootstrap_model_path = options.get("bootstrap_model_path")
        self.actor_wg = actor_wg

    @auto_await
    async def sleep_replicas(self):
        pass  # Training and inference own separate NPUs.

    wake_up_replicas = sleep_replicas
    release_kv_cache_replicas = sleep_replicas
    resume_kv_cache_replicas = sleep_replicas

    @auto_await
    async def abort_replicas(self):
        return await asyncio.to_thread(self.control.pause_generation)

    @auto_await
    async def resume_generation_replicas(self):
        return await asyncio.to_thread(self.control.resume_generation)

    def _actors(self, method, **kwargs):
        n = self.actor_wg.world_size
        return ray.get(
            self.actor_wg.execute_checkpoint_engine(
                method=[method] * n, **{k: [v] * n for k, v in kwargs.items()}
            )
        )

    def _send_weights(self, version):
        return ray.get(
            self.actor_wg.update_weights(
                global_steps=version, mode=getattr(self, "engine_backend", "elastic_hccl")
            )
        )

    @auto_await
    async def update_weights(self, global_steps=None):
        return await asyncio.to_thread(self._update, global_steps)
