"""Reuse AReaL workflow submission, but own neither routing nor vLLM processes."""

import asyncio
import os
import time
from uuid import uuid4

from areal.api import ModelResponse
from areal.engine.vllm_remote import VLLMBackend
from areal.infra import RolloutController, WorkflowExecutor
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.utils import logging, stats_tracker

from elastic_rollout.client import ControlClient
from elastic_rollout.generation import GenerationClient


class ElasticInferenceEngine(RemoteInfEngine):
    # Read by the small capability hook in AReaL RolloutController.
    externally_managed_servers = True

    def __init__(self, config):
        super().__init__(config, VLLMBackend())
        if config.use_lora or config.return_routed_experts:
            raise ValueError("only text workflows without LoRA or expert replay are supported")
        if getattr(config, "_version", "v1") != "v1":
            raise ValueError("requires AReaL v1 WorkflowExecutor/Controller")
        self.control = None
        self.generator = None

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
    ):
        if self.initialized:
            raise RuntimeError("engine already initialized")
        endpoint = addr or os.environ.get("ELASTIC_ROLLOUT_ENDPOINT")
        deployment = deployment_id or os.environ.get("ELASTIC_ROLLOUT_DEPLOYMENT")
        if not endpoint or not deployment:
            raise ValueError("control addr and deployment_id are required")
        self.control = ControlClient(endpoint, deployment, os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
        status = self.control.status()
        self._version = status["spec"]["replica"]["policy_version"]
        self.generator = GenerationClient(
            self.control,
            status["spec"]["replica"]["served_model_name"],
            request_timeout=self.config.request_timeout,
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
            raise ValueError("only text token prompts are supported")
        request = req.copy()
        if request.gconfig.n_samples != 1:
            raise ValueError("call agenerate separately for each sample (n_samples=1)")
        request.gconfig.max_new_tokens = min(
            request.gconfig.max_new_tokens,
            request.gconfig.max_tokens - len(request.input_ids),
        )
        if request.gconfig.max_new_tokens <= 0:
            raise ValueError("prompt leaves no generation token budget")
        while self.workflow_executor.is_paused():
            await asyncio.sleep(0.05)
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

    def set_version(self, version):
        if version != self._version:
            raise NotImplementedError(
                "fixed-weight inference does not support policy weight updates"
            )

    def export_stats(self):
        return stats_tracker.export_all(reduce_group=None)

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError(
            "external replicas support generation only; the runtime owns lifecycle"
        )

    launch_server = teardown_server = _unsupported
    init_weights_update_group = update_weights_from_distributed = _unsupported
    update_weights_from_disk = update_weights_from_awex = _unsupported
    pause_generation = continue_generation = offload = onload = _unsupported
    compute_logp = choose_server = set_proxy_gateway_addr = _unsupported
    _wrap_openai_agent = _unsupported
