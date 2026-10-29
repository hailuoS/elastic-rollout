"""Async framework HTTP clients retain core lease/version/cancellation semantics."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest
from test_partial_rollout import completion

from elastic_rollout.client import ControlError
from elastic_rollout.generation import GenerationClient


class Routes:
    def __init__(self):
        self.active = 0
        self.versions = []

    @asynccontextmanager
    async def route(self, key):
        self.active += 1
        version = len(self.versions) + 1
        self.versions.append(version)
        try:
            yield NS(endpoint="http://engine", serving_policy_version=version)
        finally:
            self.active -= 1


def test_async_transport_can_run_when_default_executor_is_occupied():
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        blocked = loop.run_in_executor(None, lambda: release.wait(5))
        routes = Routes()

        async def request(url, method, payload, timeout):
            assert url == "http://engine/v1/completions" and method == "POST"
            assert timeout == 2 and payload["prompt"] == [1]
            return completion([42], "stop")

        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        GenerationClient(
                            routes, "policy", request_timeout=2, request=request
                        ).generate(str(i), [1], {"max_tokens": 1})
                        for i in range(16)
                    )
                ),
                1,
            )
            assert all(item.token_ids == [42] for item in results)
            assert routes.active == 0 and len(routes.versions) == 16
        finally:
            release.set()
            await blocked

    import threading

    release = threading.Event()
    asyncio.run(run())


def test_async_transport_continuation_releases_before_resume_and_rebinds():
    routes, payloads = Routes(), []

    async def request(url, method, payload, timeout):
        payloads.append(payload)
        return completion([10]) if len(payloads) == 1 else completion([11], "stop")

    async def gate():
        assert routes.active == 0

    result = asyncio.run(
        GenerationClient(routes, "policy", request=request, wait_for_resume=gate).generate(
            "r", [1], {"max_tokens": 3}
        )
    )
    assert result.token_ids == [10, 11] and result.versions == [1, 2]
    assert payloads[1]["prompt"] == [1, 10] and payloads[1]["max_tokens"] == 2
    assert routes.active == 0


def test_async_transport_http_failure_is_not_retried_and_releases_lease():
    routes, calls = Routes(), []

    async def request(*args):
        calls.append(args)
        raise ControlError(503, "engine unavailable")

    with pytest.raises(ControlError, match="engine unavailable"):
        asyncio.run(
            GenerationClient(routes, "policy", request=request).generate(
                "r", [1], {"max_tokens": 1}
            )
        )
    assert len(calls) == 1 and routes.active == 0


def test_async_transport_cancellation_waits_for_request_before_releasing_lease():
    async def run():
        entered, finished = asyncio.Event(), asyncio.Event()
        routes = Routes()

        async def request(*args):
            entered.set()
            await finished.wait()
            return completion([42], "stop")

        task = asyncio.create_task(
            GenerationClient(routes, "policy", request=request).generate(
                "r", [1], {"max_tokens": 1}
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert routes.active == 1 and not task.done()
            finished.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert routes.active == 0
        finally:
            finished.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
