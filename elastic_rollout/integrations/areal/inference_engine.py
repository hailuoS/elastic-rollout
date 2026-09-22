"""Reuse AReaL workflow submission, but own neither routing nor vLLM processes."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING
from uuid import uuid4

from areal.api import ModelResponse
from areal.engine.vllm_remote import VLLMBackend
from areal.infra import RolloutController, WorkflowExecutor
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.utils import logging, stats_tracker

from elastic_rollout.api import nonnegative_int
from elastic_rollout.client import ControlClient, ControlError
from elastic_rollout.generation import GenerationClient

from .weight_sync import WeightSyncConfig

if TYPE_CHECKING:
    from areal.api import WeightUpdateMeta


class ElasticInferenceEngine(RemoteInfEngine):
    # Read by the small capability hook in AReaL RolloutController.
    externally_managed_servers = True
    weight_update_timeout = WeightSyncConfig().weight_update_callback_timeout_s

    def __init__(self, config):
        super().__init__(config, VLLMBackend())
        if config.use_lora or config.return_routed_experts:
            raise ValueError(
                "external inference supports text generation without LoRA or expert replay"
            )
        if getattr(config, "_version", "v1") != "v1":
            raise ValueError("MVP-0 uses AReaL v1 WorkflowExecutor/Controller")
        self.control = None
        self.generator = None
        self._weight_executor = ThreadPoolExecutor(max_workers=1)
        self._weight_lock = Lock()
        self._weight_future = None
        self._pending_update = None
        self._pending_policy_version = None
        self._policy_observer = False
        self.weight_sync_config = WeightSyncConfig()

    @classmethod
    def prepare_weight_update_meta(cls, meta: WeightUpdateMeta) -> WeightUpdateMeta:
        """Called once by the AReaL trainer before connecting training engines."""
        if meta.type != "disk" or meta.use_lora or meta.colocate_mode:
            raise ValueError("MVP-1 requires non-colocated full-model disk snapshots")
        if not meta.path:
            raise ValueError("disk snapshot path is required")
        # with_version() only replaces the basename: a unique parent survives it.
        path = Path(meta.path).parent / "elastic-rollout" / uuid4().hex / "weight_update"
        return replace(meta, path=str(path), clear_checkpoint_after_load=False)

    @classmethod
    def as_controller(cls, config, scheduler):
        return RolloutController(inf_engine=cls, config=config, scheduler=scheduler)

    def initialize(
        self,
        engine_id=None,
        addr=None,
        train_data_parallel_size=None,
        engine_rank=None,
        num_engines=None,
        deployment_id=None,
        weight_sync_config=None,
        policy_observer=False,
    ):
        if self.initialized:
            raise RuntimeError("engine already initialized")
        if type(policy_observer) is not bool:
            raise ValueError("policy_observer must be a boolean")
        self._policy_observer = policy_observer
        self.weight_sync_config = WeightSyncConfig(**(weight_sync_config or {}))
        endpoint = addr or os.environ.get("ELASTIC_ROLLOUT_ENDPOINT")
        deployment = deployment_id or os.environ.get("ELASTIC_ROLLOUT_DEPLOYMENT")
        if not endpoint or not deployment:
            raise ValueError("control addr and deployment_id are required")
        self.control = ControlClient(endpoint, deployment, os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
        status = self.control.status()
        self._version = status.get("policy", {}).get(
            "version", status["spec"]["replica"]["policy_version"]
        )
        self.generator = GenerationClient(
            self.control,
            status["spec"]["replica"]["served_model_name"],
            request_timeout=self.config.request_timeout,
            wait_for_resume=self._wait_for_workflow_resume,
        )
        self.engine_id = engine_id or uuid4().hex
        self.logger = logging.getLogger(f"[RemoteInfEngine Rank {self.engine_id}]")
        # Bind directly to this subclass; RemotevLLMEngine's wrapper would bind
        # workflows to its inner engine and bypass the external generation path.
        self.workflow_executor = WorkflowExecutor(config=self.config, inference_engine=self)
        self.workflow_executor.initialize(
            logger=self.logger,
            train_data_parallel_size=train_data_parallel_size or 1,
        )
        self._initialized = True

    async def agenerate(self, req):
        if not self.initialized:
            raise RuntimeError("initialize the elastic inference engine first")
        if req.image_data or req.vision_msg_vllm:
            raise ValueError("MVP-0 supports text token prompts only")
        request = req.copy()
        if request.gconfig.n_samples != 1:
            raise ValueError("call agenerate separately for each sample (n_samples=1)")
        request.gconfig.max_new_tokens = min(
            request.gconfig.max_new_tokens,
            request.gconfig.max_tokens - len(request.input_ids),
        )
        if request.gconfig.max_new_tokens <= 0:
            raise ValueError("prompt leaves no generation token budget")
        await self._wait_for_workflow_resume()
        http_request = self.backend.build_generation_request(
            request, with_lora=False, version=self._version
        )
        params = dict(http_request.payload)
        prompt = params.pop("prompt")
        start = time.perf_counter()
        result = await self.generator.generate(str(request.rid), prompt, params)
        latency = time.perf_counter() - start
        return ModelResponse(
            input_tokens=list(req.input_ids),
            input_images=req.image_data,
            output_tokens=result.token_ids,
            output_logprobs=result.log_probs,
            output_versions=result.versions,
            stop_reason=result.finish_reason,
            latency=latency,
            ttft=latency,
            tokenizer=req.tokenizer,
            processor=req.processor,
        )

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        if self._policy_observer:
            raise RuntimeError("policy observers cannot initiate weight updates")
        if meta.type != "disk" or not meta.path or meta.use_lora or meta.colocate_mode:
            raise ValueError("MVP-1 requires a full-model, non-colocated disk snapshot")
        if meta.clear_checkpoint_after_load:
            raise ValueError("persistent snapshots require clear_checkpoint_after_load=False")
        nonnegative_int(meta.version, "meta.version")
        with self._weight_lock:
            if meta.version <= self._version:
                raise ValueError("meta.version must exceed the current serving version")
            if self._pending_update or (self._weight_future and not self._weight_future.done()):
                raise RuntimeError("a weight update is already pending")
            # The train engine exports AFTER this method returns. Waiting here
            # would deadlock its save/barrier/name-resolve sequence.
            self._weight_future = self._weight_executor.submit(
                self._prepare_disk_update, copy(meta), self._version
            )
            return self._weight_future

    def _prepare_disk_update(self, meta: WeightUpdateMeta, base_version: int) -> None:
        from areal.utils import name_resolve, names

        name_resolve.wait(
            names.update_weights_from_disk(
                self.config.experiment_name, self.config.trial_name, base_version
            ),
            timeout=self.weight_sync_config.snapshot_export_timeout_s,
        )
        snapshot = Path(meta.path)
        if not (snapshot / "config.json").is_file() or not (
            any(snapshot.glob("*.safetensors")) or any(snapshot.glob("pytorch_model*.bin"))
        ):
            raise ValueError(f"disk export is not a standalone HF model snapshot: {meta.path}")
        # Multiple CPU workflow workers receive the same AReaL collective RPC.
        # One begins the update; the rest join that exact snapshot/base version.
        deadline = time.monotonic() + self.weight_sync_config.policy_prepare_timeout_s
        while True:
            status = self.control.status()
            if status["policy"]["version"] != base_version:
                raise RuntimeError("serving policy changed before disk update")
            try:
                update = self.control.begin_policy_update(meta.path)
                break
            except ControlError as exc:
                if exc.status != 409:
                    raise
                update = self.control.status().get("policy_update")
                if update and update["state"] != "COMMITTED":
                    if (
                        update["snapshot_path"] != meta.path
                        or update["base_version"] != base_version
                    ):
                        raise
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("deployment did not become stable for disk update") from exc
                time.sleep(0.2)
        if update["base_version"] != base_version:
            raise RuntimeError("serving policy changed before disk update")
        self.control.wait_policy_update(
            update["update_id"], timeout=max(0, deadline - time.monotonic())
        )
        with self._weight_lock:
            self._pending_update = update["update_id"]
            self._pending_policy_version = meta.version

    def set_version(self, version: int) -> None:
        nonnegative_int(version, "version")
        with self._weight_lock:
            if self._policy_observer:
                if version < self._version or self.control.status()["policy"]["version"] != version:
                    raise RuntimeError("observer version must match the committed policy")
                if version != self._version:
                    self.logger.info("policy observer following committed version %s", version)
                self._version = version
                return
            if version == self._version:
                return
            if version < self._version or not self._pending_update:
                raise RuntimeError("advancing version requires a PREPARED disk update")
            if version != self._pending_policy_version:
                raise ValueError(
                    f"version {version} does not match prepared snapshot version "
                    f"{self._pending_policy_version}"
                )
            deadline = time.monotonic() + 45
            while True:
                try:
                    self.control.commit_policy_update(self._pending_update, version)
                    break
                except ControlError as exc:
                    # A health probe may hold the manager's lifecycle lock.
                    state = self.control.get_policy_update(self._pending_update)["state"]
                    if exc.status != 409 or state not in ("PREPARED", "COMMITTED"):
                        raise
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.2)
            self._version = version
            self._pending_update = None
            self._pending_policy_version = None

    def pause_generation(self) -> None:
        self.control.pause_generation()

    def continue_generation(self) -> None:
        self.control.resume_generation()

    async def _wait_for_workflow_resume(self) -> None:
        while self.workflow_executor.is_paused():
            await asyncio.sleep(0.05)

    def destroy(self):
        self._weight_executor.shutdown(wait=True, cancel_futures=True)
        super().destroy()

    def export_stats(self):
        return stats_tracker.export_all(reduce_group=None)

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError(
            "external replicas support generation only; runtime owns physical lifecycle"
        )

    launch_server = teardown_server = _unsupported
    init_weights_update_group = update_weights_from_distributed = _unsupported
    update_weights_from_awex = _unsupported
    offload = onload = _unsupported
    compute_logp = choose_server = _unsupported


class ElasticEvaluationEngine(ElasticInferenceEngine):
    """Evaluation observes the training deployment's already committed policy."""

    def set_version(self, version: int) -> None:
        if self._policy_observer:
            return super().set_version(version)
        nonnegative_int(version, "version")
        if version == self._version:
            return
        if version < self._version or self.control.status()["policy"]["version"] != version:
            raise RuntimeError("evaluation version must match the committed training policy")
        self._version = version
