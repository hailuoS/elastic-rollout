"""Synchronous external policy transactions using verl's existing manager seam."""

import asyncio
import logging
import os
import time

import ray
from verl.utils.ray_utils import auto_await

from elastic_rollout.client import ControlClient

logger = logging.getLogger(__name__)


class ElasticCheckpointEngineManager:
    def __init__(self, config, actor_wg, replicas):
        if config.backend != "elastic_hccl" or replicas:
            raise ValueError("elastic_hccl requires external service get_replicas() == []")
        options = config.engine_kwargs["elastic_hccl"]
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

    def _update(self, version):
        started = time.monotonic()
        status = self.control.status()
        if version == 0:
            if (
                status["serving_policy_version"] != 0
                or status["bootstrap_snapshot"]
                != {"version": 0, "model_path": self.bootstrap_model_path}
                or status["policy_update"] is not None
                or not status["accepting_routes"]
            ):
                raise ValueError(
                    "initial v0 equivalence not confirmed; use matching fresh bootstrap"
                )
            return {}
        if status["inflight"]:
            raise ValueError("MVP-2A requires no active rollout before weight sync")
        update = self.control.begin_external_update(version)
        uid = update["update_id"]
        metrics = {}

        def timed(name, function, *args, **kwargs):
            before = time.monotonic()
            result = function(*args, **kwargs)
            metrics["elastic_hccl/" + name] = time.monotonic() - before
            return result

        try:
            pause_started = time.monotonic()
            timed("pause_to_inflight_zero_s", self.control.pause_generation)
            metadata = self._actors("prepare")
            if len([item for item in metadata if item]) != 1:
                raise ValueError("exactly one trainer rank0 must provide rendezvous metadata")
            master = next(item for item in metadata if item)
            session = self.control.transfer_operation(uid, "prepare", backend="hccl", **master)
            if session["needs_initialize"]:
                timed("hccl_init_s", self._actors, "configure_session", session=session)
            else:
                self._actors("configure_session", session=session)
                metrics["elastic_hccl/hccl_init_s"] = 0.0
            sid = session["session_id"]
            timed(
                "start_weight_update_s",
                self.control.transfer_operation,
                uid,
                "start",
                session_id=sid,
            )
            for result in ray.get(
                self.actor_wg.update_weights(global_steps=version, mode="elastic_hccl")
            ):
                if isinstance(result, dict):
                    metrics.update(result)
            timed("finish_verify_s", self.control.transfer_operation, uid, "finish", session_id=sid)
            timed("commit_s", self.control.commit_policy_update, uid, version)
            self.control.resume_generation()
            resumed_at = time.monotonic()
        except BaseException as exc:
            try:
                self.control.transfer_operation(uid, "fail", error=str(exc))
            except Exception:
                pass
            # No resume/rollback, including ambiguous sender/receiver failures.
            raise
        metrics["elastic_hccl/weight_sync_total_s"] = time.monotonic() - started
        metrics["elastic_hccl/logical_generation_pause_s"] = resumed_at - pause_started
        logger.info(
            "HCCL update committed update_id=%s version=%s metrics=%s", uid, version, metrics
        )
        return metrics

    @auto_await
    async def update_weights(self, global_steps=None):
        return await asyncio.to_thread(self._update, global_steps)
