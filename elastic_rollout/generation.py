"""Client-side vLLM wire codec. This module never runs as a generation proxy."""

import asyncio
import math
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .client import ControlClient, RoutingPaused, complete_on_cancel, request_json

ResumeGate = Callable[[], Awaitable[None]]


@dataclass
class Completion:
    token_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    finish_reason: str = ""


def parse_completion(response: dict) -> Completion:
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("expected exactly one completion")
    choice = choices[0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens")
    probabilities = logprobs.get("token_logprobs")
    if tokens is None or probabilities is None or len(tokens) != len(probabilities):
        raise ValueError("vLLM must return aligned token IDs and token logprobs")
    ids = []
    for token in tokens:
        if not isinstance(token, str) or not token.startswith(("token:", "token_id:")):
            raise ValueError("vLLM did not honor return_tokens_as_token_ids")
        token_id = int(token.split(":", 1)[1])
        if token_id < 0:
            raise ValueError("negative token ID")
        ids.append(token_id)
    if any(p is None or not math.isfinite(p) for p in probabilities):
        raise ValueError("invalid token logprob")
    reason = choice.get("finish_reason")
    if reason not in ("stop", "length", "abort"):
        raise ValueError(f"unsupported vLLM finish_reason: {reason}")
    return Completion(ids, probabilities, finish_reason=reason)


class GenerationClient:
    def __init__(
        self,
        control: ControlClient,
        model: str,
        request_timeout: float = 300,
        max_stalled_aborts: int = 8,
        wait_for_resume: ResumeGate | None = None,
    ):
        self.control, self.model = control, model
        if type(max_stalled_aborts) is not int or max_stalled_aborts <= 0:
            raise ValueError("max_stalled_aborts must be a positive integer")
        self.request_timeout = request_timeout
        self.max_stalled_aborts = max_stalled_aborts
        self.wait_for_resume = wait_for_resume

    @asynccontextmanager
    async def _route(self, routing_key: str):
        backoff = 0.1
        while True:
            async with AsyncExitStack() as stack:
                try:
                    lease = await stack.enter_async_context(self.control.route(routing_key))
                except RoutingPaused:
                    if self.wait_for_resume is not None:
                        await self.wait_for_resume()
                    # The framework callback may return before distributed pause
                    # state catches up. Bound polling even with no callback.
                    await asyncio.sleep(backoff)
                    backoff = min(1.0, backoff * 2)
                    continue
                # Catch only failed admission, never generation/body/release errors.
                yield lease
                return

    async def generate(
        self, routing_key: str, prompt_ids: list[int], sampling_params: dict
    ) -> Completion:
        if not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            raise ValueError("prompt_ids must contain nonnegative integer token IDs")
        params = dict(sampling_params)
        if "max_new_tokens" in params:
            if "max_tokens" in params:
                raise ValueError("specify only max_tokens or max_new_tokens")
            params["max_tokens"] = params.pop("max_new_tokens")
        budget = params.get("max_tokens")
        if type(budget) is not int or budget <= 0:
            raise ValueError("a positive max_tokens budget is required")
        if params.get("n", 1) != 1 or params.get("best_of", 1) != 1 or params.get("stream", False):
            raise ValueError("MVP-0 requires n=1, best_of=1, stream=False")
        if any(k in params for k in ("model", "prompt", "messages", "extra_body")):
            raise ValueError("sampling parameters cannot override model or prompt")
        result = Completion()
        stalled_aborts = 0
        while len(result.token_ids) < budget:
            payload = dict(
                params,
                model=self.model,
                prompt=list(prompt_ids) + result.token_ids,
                max_tokens=budget - len(result.token_ids),
                stream=False,
                logprobs=0,
                return_tokens_as_token_ids=True,
            )
            async with self._route(routing_key) as lease:
                response = await complete_on_cancel(
                    asyncio.to_thread(
                        request_json,
                        lease.endpoint + "/v1/completions",
                        "POST",
                        payload,
                        self.request_timeout,
                    )
                )
                segment = parse_completion(response)
                if len(result.token_ids) + len(segment.token_ids) > budget:
                    raise ValueError("vLLM exceeded the remaining token budget")
                result.token_ids.extend(segment.token_ids)
                result.log_probs.extend(segment.log_probs)
                result.versions.extend([lease.serving_policy_version] * len(segment.token_ids))
                result.finish_reason = segment.finish_reason
            if result.finish_reason != "abort":
                return result
            if len(result.token_ids) >= budget:
                result.finish_reason = "length"
                return result
            stalled_aborts = 0 if segment.token_ids else stalled_aborts + 1
            if stalled_aborts >= self.max_stalled_aborts:
                raise RuntimeError("generation repeatedly aborted without token progress")
            # Release the segment lease before waiting. Framework resume happens
            # after policy commit; every continuation acquires a fresh route.
            if self.wait_for_resume is not None:
                await self.wait_for_resume()
            else:
                await asyncio.sleep(0.1)
