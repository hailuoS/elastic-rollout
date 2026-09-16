from typing import Any

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from elastic_rollout.client import ControlClient
from elastic_rollout.generation import GenerationClient


class ElasticLLMClient(LLMServerClient):
    def __init__(self, config, control: ControlClient, model: str, request_timeout: float = 300):
        super().__init__(config=config, load_balancer_handle=None)
        self.generator = GenerationClient(control, model, request_timeout)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        **kwargs,
    ) -> TokenOutput:
        if any(x is not None for x in (image_data, video_data, audio_data)) or mm_processor_kwargs:
            raise ValueError("MVP-0 supports text token prompts only")
        params = dict(sampling_params)
        if "max_tokens" not in params and "max_new_tokens" not in params:
            params["max_tokens"] = self.config.actor_rollout_ref.rollout.response_length
        priority = kwargs.pop("priority", 0)
        if priority:
            params["priority"] = priority
        if kwargs:
            raise ValueError(f"unsupported generation options: {sorted(kwargs)}")
        output = await self.generator.generate(str(request_id), list(prompt_ids), params)
        versions = output.versions
        return TokenOutput(
            token_ids=output.token_ids,
            log_probs=output.log_probs,
            stop_reason="completed",
            num_preempted=0,
            extra_fields={
                "global_steps": versions[-1] if versions else None,
                "min_global_steps": min(versions) if versions else None,
                "max_global_steps": max(versions) if versions else None,
                "serving_policy_versions": versions,
                "finish_reason": output.finish_reason,
            },
        )
