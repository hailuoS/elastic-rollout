"""External rollout service provider, independent of experimental fully-async managers."""

import asyncio
import os

from verl.utils.ray_utils import auto_await

from elastic_rollout.client import ControlClient

from .llm_client import ElasticLLMClient


class ElasticRolloutService:
    def __init__(self, config, worker_group=None, rollout_resource_pool=None, **kwargs):
        if worker_group is not None or rollout_resource_pool is not None:
            raise ValueError(
                "standalone AgentLoop generation requires no trainer "
                "worker_group/resource_pool; PPO weight synchronization is unsupported"
            )
        if kwargs:
            raise ValueError(f"unsupported service options: {sorted(kwargs)}")
        self.config = config
        custom = config.actor_rollout_ref.rollout.get("custom") or {}
        options = custom.get("elastic", {})
        if options.get("fixed_policy") is not True:
            raise ValueError("set rollout.custom.elastic.fixed_policy=true for generation")
        self.control = ControlClient(
            options["endpoint"],
            options["deployment_id"],
            os.environ.get("ELASTIC_ROLLOUT_TOKEN"),
        )
        self.request_timeout = options.get("request_timeout", 300)
        self.model = None

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        service = cls(*args, **kwargs)
        status = await asyncio.to_thread(service.control.status)
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
        raise NotImplementedError(
            "External replicas are not verl RolloutReplica objects. "
            "CheckpointEngine weight synchronization is unsupported."
        )

    def destroy(self):
        """The runtime outlives framework clients and owns replica cleanup."""
