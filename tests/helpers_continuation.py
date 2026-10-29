"""HTTP continuation scenario shared by core behavioral tests."""

import asyncio
import time
from collections import Counter

from elastic_rollout.client import ControlClient, ControlError
from elastic_rollout.generation import GenerationClient


def stable_call(call, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return call()
        except ControlError as exc:
            if exc.status != 409 or time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


async def run(
    control: ControlClient,
    prompt_ids: list[int],
    *,
    max_tokens: int = 2048,
    abort_after: float = 1,
    snapshot: str | None = None,
    version: int | None = None,
    timeout: float = 900,
) -> dict:
    if (snapshot is None) != (version is None):
        raise ValueError("snapshot and version must be supplied together")
    initial = await asyncio.to_thread(control.status)
    if (
        not initial["accepting_routes"]
        or initial["inflight"]
        or initial["desired_replicas"] <= 0
        or initial["ready_replicas"] != initial["desired_replicas"]
        or initial["total_replicas"] != initial["desired_replicas"]
    ):
        raise ValueError("use a stable, idle, dedicated deployment with routing open")
    base = initial["policy"]["version"]
    if version is not None and version <= base:
        raise ValueError("version must advance the committed policy")
    aborted, resume = asyncio.Event(), asyncio.Event()

    async def resume_gate():
        # GenerationClient enters this callback only after parsing abort and releasing.
        assert (await asyncio.to_thread(control.status))["inflight"] == 0
        aborted.set()
        await resume.wait()

    model = initial["spec"]["replica"]["served_model_name"]
    generator = GenerationClient(
        control, model, request_timeout=timeout, wait_for_resume=resume_gate
    )
    task = asyncio.create_task(
        generator.generate(
            "partial-rollout-smoke",
            prompt_ids,
            {"max_tokens": max_tokens, "temperature": 0, "ignore_eos": True},
        )
    )
    waiter = asyncio.create_task(aborted.wait())
    try:

        async def wait_inflight():
            while not (await asyncio.to_thread(control.status))["inflight"]:
                if task.done():
                    await task
                    raise RuntimeError("generation finished before pause; increase max_tokens")
                await asyncio.sleep(0.05)

        await asyncio.wait_for(wait_inflight(), timeout)
        await asyncio.sleep(abort_after)
        await asyncio.to_thread(control.pause_generation)
        done, _ = await asyncio.wait(
            (task, waiter), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if waiter not in done:
            if task in done:
                await task
            raise RuntimeError("no partial abort returned; inspect vLLM API and request timing")
        assert not task.done()
        status = await asyncio.to_thread(control.status)
        assert status["generation_paused"] and status["inflight"] == 0
        update = None
        if snapshot is not None:
            update = await asyncio.to_thread(
                stable_call, lambda: control.begin_policy_update(snapshot)
            )
            await asyncio.to_thread(
                control.wait_policy_update, update["update_id"], timeout=timeout
            )
        await asyncio.to_thread(control.resume_generation)
        if update is not None:
            assert not (await asyncio.to_thread(control.status))["accepting_routes"]
            await asyncio.to_thread(
                stable_call,
                lambda: control.commit_policy_update(update["update_id"], version),
            )
        resume.set()
        result = await asyncio.wait_for(task, timeout)
        assert len(result.token_ids) == len(result.log_probs) == len(result.versions)
        assert set(result.versions) == ({base, version} if update else {base}), (
            "expected nonempty partial output on both policies; tune --abort-after/--max-tokens"
        )
        fresh = await GenerationClient(control, model).generate(
            "after-resume-smoke",
            prompt_ids,
            {"max_tokens": 1, "temperature": 0},
        )
        assert fresh.versions == [version if update else base]
        assert (await asyncio.to_thread(control.status))["inflight"] == 0
        return {
            "finish_reason": result.finish_reason,
            "tokens": len(result.token_ids),
            "version_token_counts": dict(Counter(result.versions)),
            "inflight": 0,
        }
    finally:
        waiter.cancel()
        task.cancel()
        await asyncio.gather(task, waiter, return_exceptions=True)
