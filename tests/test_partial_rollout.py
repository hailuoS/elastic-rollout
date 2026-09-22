"""Partial continuation uses a released lease, framework barrier and fresh policy route."""

import asyncio
import copy
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_policy_updates import prepare

from elastic_rollout.client import ControlError
from elastic_rollout.engine.vllm import StandardVLLMControl
from elastic_rollout.generation import GenerationClient
from elastic_rollout.runtime import Reconciler


def completion(tokens, reason="abort"):
    return {
        "choices": [
            {
                "finish_reason": reason,
                "logprobs": {
                    "tokens": [f"token_id:{t}" for t in tokens],
                    "token_logprobs": [-0.1] * len(tokens),
                },
            }
        ]
    }


class Routes:
    def __init__(self):
        self.acquired = []
        self.released = []

    @asynccontextmanager
    async def route(self, key):
        self.acquired.append(key)
        try:
            yield NS(endpoint="http://vllm", serving_policy_version=len(self.acquired))
        finally:
            self.released.append(key)


def test_cross_version_rebind_waits_after_release(http_runtime, monkeypatch):
    client, runtime = http_runtime
    controller, manager, driver, _ = runtime
    manager.committed_policy = replace(manager.committed_policy, version=3)
    for replica in manager.traffic.replicas.values():
        replica.serving_policy_version = 3
    acquire = MagicMock(wraps=client.acquire)
    monkeypatch.setattr(client, "acquire", acquire)
    requests = []

    def generate(url, method, payload, timeout):
        requests.append((url, payload))
        return completion([10, 11]) if len(requests) == 1 else completion([12, 13], "stop")

    monkeypatch.setattr("elastic_rollout.generation.request_json", generate)

    async def run():
        waiting, resume = asyncio.Event(), asyncio.Event()

        async def gate():
            assert not manager.traffic.leases
            waiting.set()
            await resume.wait()

        generator = GenerationClient(client, "policy", wait_for_resume=gate)
        task = asyncio.create_task(generator.generate("logical", [1, 2], {"max_tokens": 6}))
        try:
            await asyncio.wait_for(waiting.wait(), 5)
            assert acquire.call_count == 1 and not task.done()
            assert client.status()["inflight"] == 0
            client.pause_generation()
            update = prepare(runtime)
            client.resume_generation()
            assert not client.status()["accepting_routes"]
            controller.commit_policy_update("test", update, 4)
            # Occupy the previous replica; least-inflight routing picks another.
            load = client.acquire("other-work")
            assert load.endpoint + "/v1/completions" == requests[0][0]
            assert acquire.call_count == 2  # Only explicit other-work; no continuation yet.
            resume.set()
            result = await asyncio.wait_for(task, 5)
            client.release(load.lease_id)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert result.token_ids == [10, 11, 12, 13]
        assert result.log_probs == [-0.1] * 4
        assert result.versions == [3, 3, 4, 4]
        assert result.finish_reason == "stop"

    asyncio.run(run())
    assert requests[0][0] != requests[1][0]
    assert requests[1][1]["prompt"] == [1, 2, 10, 11]
    assert requests[1][1]["max_tokens"] == 4
    assert [c.args[0] for c in acquire.call_args_list] == ["logical", "other-work", "logical"]
    assert client.status()["inflight"] == 0


def test_progressing_aborts_have_no_segment_limit(monkeypatch):
    routes = Routes()
    request = MagicMock(side_effect=[completion([i]) for i in range(12)])
    monkeypatch.setattr("elastic_rollout.generation.request_json", request)
    gate = AsyncMock()
    result = asyncio.run(
        GenerationClient(routes, "policy", wait_for_resume=gate).generate(
            "logical",
            [99],
            {"max_tokens": 12},
        )
    )
    assert result.token_ids == list(range(12))
    assert result.versions == list(range(1, 13))
    assert result.finish_reason == "length"
    assert gate.await_count == 11  # Budget exhausted on abort: no extra route or wait.
    assert len(routes.acquired) == len(routes.released) == 12
    for index, call in enumerate(request.call_args_list):
        assert call.args[2]["prompt"] == [99] + list(range(index))
        assert call.args[2]["max_tokens"] == 12 - index


def test_zero_progress_guard_resets_after_progress(monkeypatch):
    routes = Routes()
    request = MagicMock(
        side_effect=[
            completion([]),
            completion([10]),
            completion([]),
            completion([]),
        ]
    )
    monkeypatch.setattr("elastic_rollout.generation.request_json", request)
    gate = AsyncMock()
    with pytest.raises(RuntimeError, match="without token progress"):
        asyncio.run(
            GenerationClient(
                routes,
                "policy",
                max_stalled_aborts=2,
                wait_for_resume=gate,
            ).generate("logical", [99], {"max_tokens": 8})
        )
    assert len(routes.acquired) == len(routes.released) == 4
    assert gate.await_count == 3


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("crash"),
        TimeoutError("timeout"),
        ControlError(500, "server failed"),
        ValueError("malformed response"),
    ],
)
def test_transport_and_parse_errors_never_replay(monkeypatch, error):
    routes, gate = Routes(), AsyncMock()
    # Error after a valid partial segment is still not grounds to replay it.
    monkeypatch.setattr(
        "elastic_rollout.generation.request_json",
        MagicMock(
            side_effect=[
                completion([10]),
                error,
            ]
        ),
    )
    with pytest.raises(type(error), match=str(error)):
        asyncio.run(
            GenerationClient(routes, "policy", wait_for_resume=gate).generate(
                "logical",
                [99],
                {"max_tokens": 8},
            )
        )
    assert len(routes.acquired) == len(routes.released) == 2
    assert gate.await_count == 1


def test_malformed_abort_is_not_a_continuation(monkeypatch):
    routes = Routes()
    monkeypatch.setattr(
        "elastic_rollout.generation.request_json",
        lambda *a: {
            "choices": [{"finish_reason": "abort", "logprobs": None}],
        },
    )
    with pytest.raises(ValueError, match="aligned"):
        asyncio.run(GenerationClient(routes, "policy").generate("key", [1], {"max_tokens": 3}))
    assert len(routes.acquired) == len(routes.released) == 1


def test_cancellation_at_resume_gate_has_no_lease(monkeypatch):
    routes = Routes()
    monkeypatch.setattr("elastic_rollout.generation.request_json", lambda *a: completion([10]))

    async def run():
        entered = asyncio.Event()

        async def gate():
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            GenerationClient(routes, "policy", wait_for_resume=gate).generate(
                "logical",
                [99],
                {"max_tokens": 8},
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert len(routes.acquired) == len(routes.released) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(routes.acquired) == 1

    asyncio.run(run())


def test_adapter_mid_request_pause_and_deployment_control(areal_adapter, monkeypatch):
    module, executor = areal_adapter
    engine = module.ElasticInferenceEngine(
        NS(
            use_lora=False,
            return_routed_experts=False,
            request_timeout=30,
        )
    )
    engine.initialize(addr="http://control", deployment_id="test", policy_observer=True)
    engine.pause_generation()
    engine.continue_generation()
    engine.control.pause_generation.assert_called_once_with()
    engine.control.resume_generation.assert_called_once_with()
    engine.control.replicas.assert_not_called()
    routes = Routes()
    engine.generator.control = routes
    started, finish_first = threading.Event(), threading.Event()

    def generate(*args):
        if len(routes.acquired) == 1:
            started.set()
            assert finish_first.wait(5)
            return completion([10])
        return completion([11], "stop")

    monkeypatch.setattr("elastic_rollout.generation.request_json", generate)
    request = NS(
        rid="logical",
        input_ids=[1, 2],
        image_data=[],
        vision_msg_vllm=None,
        tokenizer=None,
        processor=None,
        gconfig=NS(
            n_samples=1,
            max_tokens=8,
            max_new_tokens=6,
            top_p=1,
            top_k=-1,
            greedy=True,
            temperature=1,
            stop_token_ids=[],
            ignore_eos=False,
            skip_special_tokens=True,
            frequency_penalty=0,
            use_beam_search=False,
            stop=None,
        ),
    )
    request.copy = lambda: copy.deepcopy(request)

    async def until(predicate):
        while not predicate():
            await asyncio.sleep(0.001)

    async def run():
        task = asyncio.create_task(engine.agenerate(request))
        try:
            await asyncio.wait_for(until(started.is_set), 5)
            executor.return_value.is_paused.return_value = True
            finish_first.set()
            await asyncio.wait_for(until(lambda: bool(routes.released)), 5)
            # Exercise multiple polls of the actual adapter's workflow resume gate.
            await asyncio.sleep(0.12)
            assert not task.done() and len(routes.acquired) == 1
            executor.return_value.is_paused.return_value = False
            result = await asyncio.wait_for(task, 5)
            assert result.output_tokens == [10, 11]
            assert result.output_versions == [1, 2]  # Lease versions, not adapter's local 7.
        finally:
            finish_first.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        engine._weight_executor.shutdown()
    engine.control.replicas.assert_not_called()
    assert len(routes.released) == 2


@pytest.mark.parametrize("update_policy", [False, True])
def test_smoke_uses_real_http_abort_resume_and_continuation(
    http_runtime, monkeypatch, update_policy
):
    from examples.partial_rollout_smoke import run

    client, runtime = http_runtime
    aborted = threading.Event()
    generation_started = threading.Event()
    physical_paused = threading.Event()
    requests, controls = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.path == "/is_paused"
            body = json.dumps({"is_paused": physical_paused.is_set()}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/pause?mode=abort":
                controls.append(self.path)
                physical_paused.set()
                aborted.set()
                response = {"status": "paused"}
            elif self.path == "/resume":
                controls.append(self.path)
                physical_paused.clear()
                response = {"status": "resumed"}
            else:
                assert self.path == "/v1/completions"
                assert not physical_paused.is_set()
                requests.append(payload)
                if len(requests) == 1:
                    generation_started.set()
                    assert aborted.wait(5)
                    response = completion([10, 11])
                elif len(requests) == 2:
                    response = completion([12, 13], "stop")
                else:
                    response = completion([14], "length")
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for replica in runtime[1].traffic.replicas.values():
        replica.endpoint = f"http://127.0.0.1:{server.server_port}"
    control = StandardVLLMControl()

    def pause(replica, **kwargs):
        assert generation_started.wait(5)
        control.pause_generation(replica.endpoint, **kwargs)

    monkeypatch.setattr(runtime[2], "pause_generation", pause)
    monkeypatch.setattr(
        runtime[2], "resume_generation", lambda r, **kw: control.resume_generation(r.endpoint, **kw)
    )
    reconciler = Reconciler(runtime[0], interval=0.01)
    reconciler.start()
    try:
        result = asyncio.run(
            run(
                client,
                [1, 2],
                max_tokens=6,
                abort_after=0.05,
                timeout=5,
                snapshot="/shared/policy-1" if update_policy else None,
                version=1 if update_policy else None,
            )
        )
        assert result["version_token_counts"] == ({0: 2, 1: 2} if update_policy else {0: 4})
        assert result["inflight"] == 0
        assert requests[1]["prompt"] == [1, 2, 10, 11]
        assert requests[1]["max_tokens"] == 4
        assert controls[-2:] == ["/resume"] * 2
        assert len(controls) >= 4 and all(p == "/pause?mode=abort" for p in controls[:-2])
    finally:
        aborted.set()
        reconciler.stop()
        server.shutdown()
        server.server_close()
        thread.join()
