"""Real HTTP regression for default framework transports and session lifetime."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest
from test_integrations import PACKAGE, load_adapter, stub_module

from elastic_rollout.client import ControlError
from elastic_rollout.integrations.http import AsyncGenerationClient

aiohttp = pytest.importorskip("aiohttp")


class Routes:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.active = 0

    @asynccontextmanager
    async def route(self, key):
        self.active += 1
        try:
            yield NS(endpoint=self.endpoint, serving_policy_version=3)
        finally:
            self.active -= 1


class Engine:
    def __init__(self, count=1, status=200, body=None, reject_first=False):
        self.count, self.status, self.body = count, status, body
        self.reject_first = reject_first
        self.arrived = 0
        self.ready = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.active = 0

    async def handle(self, reader, writer):
        entered = False
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.splitlines()
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            self.arrived += 1
            number = self.arrived
            self.active += 1
            self.finished.clear()
            entered = True
            if self.arrived == self.count:
                self.ready.set()
            reject = self.reject_first and number == 1
            if not reject:
                await self.release.wait()
            status = 503 if reject else self.status
            body = (
                self.body
                if self.body is not None
                else json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "logprobs": {"tokens": ["token_id:42"], "token_logprobs": [-0.1]},
                            }
                        ]
                    }
                ).encode()
            )
            writer.write(
                f"HTTP/1.1 {status} Response\r\nContent-Length: {len(body)}\r\n"
                "Location: /must-not-replay\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if entered:
                self.active -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            if self.active == 0:
                self.finished.set()


@asynccontextmanager
async def engine_server(engine):
    server = await asyncio.start_server(engine.handle, "127.0.0.1", 0, backlog=256)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        engine.release.set()
        if engine.active:
            await asyncio.wait_for(engine.finished.wait(), 3)
        server.close()
        await server.wait_closed()


@pytest.fixture
def sessions(monkeypatch):
    original = aiohttp.ClientSession
    created = []

    def create(*args, **kwargs):
        session = original(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", create)
    return created


@pytest.mark.parametrize("framework", ["areal", "verl"])
def test_framework_default_sends_128_requests_without_generation_threads(
    framework, http_runtime, areal_adapter, monkeypatch, sessions
):
    control, runtime = http_runtime
    if framework == "areal":
        module, _ = areal_adapter
        monkeypatch.setattr(module, "ControlClient", lambda *a, **kw: control)
        owner = module.ElasticInferenceEngine(
            NS(use_lora=False, return_routed_experts=False, request_timeout=10)
        )
        owner.initialize(addr=control.endpoint, deployment_id="test")
    else:

        class Base:
            def __init__(self, **kwargs):
                self.config = kwargs["config"]

        stub_module(monkeypatch, "verl")
        stub_module(monkeypatch, "verl.workers")
        stub_module(monkeypatch, "verl.workers.rollout")
        stub_module(monkeypatch, "verl.workers.rollout.llm_server", LLMServerClient=Base)
        stub_module(monkeypatch, "verl.workers.rollout.replica", TokenOutput=NS)
        module = load_adapter(
            monkeypatch,
            "elastic_rollout.integrations.verl.llm_client",
            PACKAGE / "integrations/verl/llm_client.py",
        )
        owner = module.ElasticLLMClient(NS(), control, "policy", request_timeout=10)
    generator = owner.generator
    assert isinstance(generator, AsyncGenerationClient)

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        unblock = threading.Event()
        blocker = loop.run_in_executor(None, lambda: unblock.wait(10))
        engine = Engine(count=128)
        async with engine_server(engine) as endpoint:
            for replica in runtime[1].traffic.replicas.values():
                replica.endpoint = endpoint
            tasks = [
                asyncio.create_task(generator.generate(str(i), [1], {"max_tokens": 1}))
                for i in range(128)
            ]
            try:
                await asyncio.wait_for(engine.ready.wait(), 5)
                assert control.status()["inflight"] == 128
                assert len(sessions) == 1  # One shared pool, not one session per request.
                engine.release.set()
                results = await asyncio.wait_for(asyncio.gather(*tasks), 5)
                assert all(result.token_ids == [42] for result in results)
                assert control.status()["inflight"] == 0
                assert sessions[0].closed
            finally:
                unblock.set()
                engine.release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                await blocker

    try:
        asyncio.run(run())
    finally:
        if framework == "areal":
            owner._weight_executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("cancel_count", [1, 3])
def test_cancellation_keeps_lease_and_pool_until_http_finishes(sessions, cancel_count):
    async def run():
        engine = Engine()
        async with engine_server(engine) as endpoint:
            routes = Routes(endpoint)
            client = AsyncGenerationClient(routes, "policy", request_timeout=2)
            task = asyncio.create_task(client.generate("cancel", [1], {"max_tokens": 1}))
            try:
                await asyncio.wait_for(engine.ready.wait(), 2)
                for _ in range(cancel_count):
                    task.cancel()
                    await asyncio.sleep(0.01)
                assert routes.active == 1 and not task.done() and not sessions[0].closed
                engine.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert routes.active == 0 and sessions[0].closed
            finally:
                engine.release.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("status", [302, 503])
def test_generation_errors_and_redirects_are_not_replayed(status, sessions):
    async def run():
        engine = Engine(status=status, body=b'{"error":"reject"}')
        engine.release.set()
        async with engine_server(engine) as endpoint:
            routes = Routes(endpoint)
            with pytest.raises(ControlError) as exc:
                await AsyncGenerationClient(routes, "policy").generate("r", [1], {"max_tokens": 1})
            assert exc.value.status == status
            assert engine.arrived == 1 and routes.active == 0 and sessions[0].closed

    asyncio.run(run())


def test_client_can_be_reused_across_event_loops_without_stale_sessions(sessions):
    routes = Routes("")
    client = AsyncGenerationClient(routes, "policy")

    async def run():
        engine = Engine()
        engine.release.set()
        async with engine_server(engine) as endpoint:
            routes.endpoint = endpoint
            assert (await client.generate("r", [1], {"max_tokens": 1})).token_ids == [42]

    asyncio.run(run())
    asyncio.run(run())
    assert len(sessions) == 2 and all(session.closed for session in sessions)


def test_timeout_closes_pool_and_releases_lease(sessions):
    async def run():
        engine = Engine()
        async with engine_server(engine) as endpoint:
            routes = Routes(endpoint)
            with pytest.raises(asyncio.TimeoutError):
                await AsyncGenerationClient(routes, "policy", request_timeout=0.05).generate(
                    "r", [1], {"max_tokens": 1}
                )
            assert routes.active == 0 and sessions[0].closed

    asyncio.run(run())


def test_one_failed_generation_does_not_close_another_requests_pool(sessions):
    async def run():
        engine = Engine(count=2, reject_first=True)
        async with engine_server(engine) as endpoint:
            routes = Routes(endpoint)
            client = AsyncGenerationClient(routes, "policy", request_timeout=3)
            tasks = [
                asyncio.create_task(client.generate(str(i), [1], {"max_tokens": 1}))
                for i in range(2)
            ]
            try:
                await asyncio.wait_for(engine.ready.wait(), 2)
                done, pending = await asyncio.wait(
                    tasks, timeout=2, return_when=asyncio.FIRST_COMPLETED
                )
                assert len(done) == len(pending) == 1
                with pytest.raises(ControlError):
                    next(iter(done)).result()
                assert len(sessions) == 1 and not sessions[0].closed and routes.active == 1
                engine.release.set()
                result = await next(iter(pending))
                assert result.token_ids == [42] and routes.active == 0
                assert sessions[0].closed
            finally:
                engine.release.set()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())
