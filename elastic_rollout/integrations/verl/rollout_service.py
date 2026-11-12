"""External rollout service provider for verl Trainer V1."""

import asyncio
import os

from verl.utils.ray_utils import auto_await

from elastic_rollout.client import ControlClient

from .llm_client import ElasticLLMClient


class ElasticRolloutService:
    def __init__(
        self, config, worker_group=None, rollout_resource_pool=None, start_rank=0, **kwargs
    ):
        if kwargs:
            raise ValueError(f"unsupported service options: {sorted(kwargs)}")
        self.config = config
        custom = config.actor_rollout_ref.rollout.get("custom") or {}
        options = custom.get("elastic", {})
        self.training = options.get("fixed_policy") is False and options.get("weight_sync") in (
            "hccl",
            "nccl",
        )
        self.backend = options.get("weight_sync", "hccl")
        if not self.training and (worker_group is not None or rollout_resource_pool is not None):
            raise ValueError("fixed-policy mode requires a standalone generation entrypoint")
        if not self.training and options.get("fixed_policy") is not True:
            raise ValueError("set rollout.custom.elastic.fixed_policy=true for MVP-0 generation")
        if self.training:
            rollout = config.actor_rollout_ref.rollout
            if (
                rollout.get("checkpoint_manager_class")
                != (
                    "elastic_rollout.integrations.verl.checkpoint_manager."
                    "ElasticCheckpointEngineManager"
                )
                or rollout.checkpoint_engine.backend != f"elastic_{self.backend}"
            ):
                raise ValueError(
                    "training requires ElasticCheckpointEngineManager and elastic_hccl"
                )
            if config.get("async_training"):
                raise ValueError("experimental fully-async training is not supported")
            if config.trainer.use_v1:
                if config.trainer.v1.trainer_mode != "separate_async":
                    raise ValueError("Elastic MVP-2B requires V1 separate_async trainer")
                if worker_group is not None or rollout_resource_pool is not None:
                    raise ValueError("V1 separate_async requires a standalone rollout entrypoint")
                if config.actor_rollout_ref.hybrid_engine:
                    raise ValueError("Elastic MVP-2B requires hybrid_engine=false")
                if rollout.multi_turn.enable:
                    raise ValueError("Elastic MVP-2B initial acceptance requires multi_turn=false")
                if config.get("global_profiler") and config.global_profiler.get("steps"):
                    raise ValueError(
                        "Elastic MVP-2B initial acceptance requires profiling disabled"
                    )
            if rollout.get("free_cache_engine", False):
                raise ValueError("external HCCL requires free_cache_engine=false; separate NPUs")
            engine_options = rollout.checkpoint_engine.engine_kwargs[f"elastic_{self.backend}"]
            if (
                engine_options["endpoint"] != options["endpoint"]
                or engine_options["deployment_id"] != options["deployment_id"]
                or engine_options.get("bootstrap_model_path") != config.actor_rollout_ref.model.path
            ):
                raise ValueError("generation/checkpoint endpoint and trainer bootstrap must match")
        self.control = ControlClient(
            options["endpoint"],
            options["deployment_id"],
            os.environ.get("ELASTIC_ROLLOUT_TOKEN"),
        )
        self.request_timeout = options.get("request_timeout")
        self.model = None

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        service = cls(*args, **kwargs)
        status = await asyncio.to_thread(service.control.status)
        if service.training:
            if status["spec"]["replica"]["weight_sync"] != service.backend or status[
                "bootstrap_snapshot"
            ]["model_path"] != config_path(service.config):
                raise ValueError("Core must use HCCL and the same trainer bootstrap model")
        service.model = status["spec"]["replica"]["served_model_name"]
        return service

    def get_client(self, client_cls=None, **kwargs):
        if client_cls is not None or kwargs:
            raise ValueError("ElasticRolloutService supplies its own external-routing client")
        return ElasticLLMClient(self.config, self.control, self.model, self.request_timeout)

    def get_addresses(self):
        # For observation only. Clients always AcquireRoute, never cache this list.
        return [r["endpoint"] for r in self.control.replicas() if r["state"] == "READY"]

    def get_replicas(self):
        if not self.training:
            raise NotImplementedError("fixed-policy service has no CheckpointEngine integration")
        return []  # Deliberately no fake verl RolloutReplica objects.

    def get_active_server_count(self):
        return self.control.status()["ready_replicas"]

    def get_n_gpus(self):
        # Retain draining/failed assignments until their resource lease is released.
        return sum(
            len(r["allocation"]["actual_device_ids"])
            for r in self.control.replicas()
            if r["resource"] is not None and r["allocation"] is not None
        )

    def destroy(self):
        """The runtime outlives framework clients and owns replica cleanup."""


def config_path(config):
    return config.actor_rollout_ref.model.path
