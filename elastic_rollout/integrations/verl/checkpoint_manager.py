"""Synchronous external policy transactions using verl's existing manager seam."""

import asyncio
import json
import logging
import os
import time

import ray
from verl.utils.ray_utils import auto_await

from elastic_rollout.client import ControlClient, ControlError

logger = logging.getLogger(__name__)
_LIFECYCLE_RETRY_WAIT_BUDGET_S = 10.0


def _is_retryable_lifecycle_conflict(exc: ControlError) -> bool:
    if exc.status != 409:
        return False
    try:
        payload = json.loads(str(exc))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("error") == "deployment lifecycle mutation in progress"
    )


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

    def _begin_external_update_with_retry(self, version):
        # Only rejected lifecycle-lock acquisitions are safe to retry. This is
        # cumulative sleep time; individual HTTP calls have their own timeout.
        remaining_wait = _LIFECYCLE_RETRY_WAIT_BUDGET_S
        delay = 0.1
        retries = 0
        while True:
            try:
                update = self.control.begin_external_update(version)
                if retries:
                    logger.info(
                        "external update began after lifecycle contention version=%s retries=%s",
                        version,
                        retries,
                    )
                return update
            except ControlError as exc:
                if not _is_retryable_lifecycle_conflict(exc) or remaining_wait <= 0:
                    raise
                if retries == 0:
                    logger.warning(
                        "deployment lifecycle busy before external update version=%s; retrying",
                        version,
                    )
                sleep_for = min(delay, remaining_wait)
                before_sleep = time.monotonic()
                time.sleep(sleep_for)
                remaining_wait -= max(sleep_for, time.monotonic() - before_sleep)
                delay = min(delay * 2, 0.5)
                retries += 1

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
        update = self._begin_external_update_with_retry(version)
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
