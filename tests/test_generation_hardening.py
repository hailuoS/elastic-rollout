"""vLLM 0.23 quiescence, explicit admission rejection, and bounded control budgets."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_partial_rollout import completion
from test_policy_updates import prepare
from test_runtime import ready

from elastic_rollout.api import DeploymentSpec, ReplicaSpec
from elastic_rollout.client import ControlClient, ControlError, RoutingPaused
from elastic_rollout.engine.vllm import StandardVLLMControl
from elastic_rollout.generation import GenerationClient
from elastic_rollout.runtime import GenerationControlTimeout


def test_late_admission_requires_second_abort_before_pause_success(http_runtime, monkeypatch):
    client, runtime = http_runtime
    controller, manager, driver, _ = runtime
    queued, first_pause, arrived, aborted = [threading.Event() for _ in range(4)]
    target = next(iter(manager.traffic.replicas))
    rounds = 0
    requests = []

    def generation(*args):
        requests.append(args)
        if len(requests) == 1:
            queued.set()  # Acquire succeeded; HTTP has not yet reached vLLM.
            assert first_pause.wait(5)
            arrived.set()  # Late request is queued in the already paused engine.
            assert aborted.wait(5)
            return completion([])
        return completion([42], "stop")

    def pause(replica, **kwargs):
        nonlocal rounds
        driver.paused.add(replica.replica_id)
        if replica.replica_id == target:
            rounds += 1
            if rounds == 1:
                first_pause.set()
                assert arrived.wait(5)
            else:
                aborted.set()

    monkeypatch.setattr("elastic_rollout.generation.request_json", generation)
    monkeypatch.setattr(driver, "pause_generation", pause)

    async def run():
        resumed = asyncio.Event()
        generator = GenerationClient(client, "policy", wait_for_resume=resumed.wait)
        task = asyncio.create_task(generator.generate("late", [1], {"max_tokens": 4}))
        try:
            assert await asyncio.to_thread(queued.wait, 5)
            status = await asyncio.wait_for(asyncio.to_thread(client.pause_generation), 5)
            assert status["inflight"] == 0 and status["generation_paused"]
            assert status["route_pause_reasons"] == ["generation"]
            assert rounds >= 2 and not task.done()
            update = prepare(runtime)  # No circular wait between inflight and preparation.
            client.resume_generation()
            controller.commit_policy_update("test", update, 1)
            resumed.set()
            result = await asyncio.wait_for(task, 5)
            assert result.token_ids == [42] and result.versions == [1]
            assert len(requests) == 2
        finally:
            first_pause.set()
            aborted.set()
            resumed.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("lock_busy", [False, True])
def test_pause_deadline_fails_closed_with_context(http_runtime, lock_busy):
    client, runtime = http_runtime
    manager = runtime[1]
    manager.spec = replace(manager.spec, generation_control_timeout_s=0.05)
    lease = client.acquire("never-drains")
    if lock_busy:
        manager.generation_lock.acquire()
    try:
        with pytest.raises(ControlError) as error:
            client.pause_generation()
    finally:
        if lock_busy:
            manager.generation_lock.release()
    detail = json.loads(str(error.value))
    assert error.value.status == 504 and detail["code"] == "GENERATION_CONTROL_TIMEOUT"
    assert "inflight=1" in detail["error"] and lease.replica_id in detail["error"]
    assert "0.05s" in detail["error"]
    with pytest.raises(RoutingPaused):
        client.acquire("blocked")
    assert not client.status()["accepting_routes"]
    assert "generation" in client.status()["route_pause_reasons"]
    client.release(lease.lease_id)


def test_sequential_targets_share_one_deadline(runtime, monkeypatch):
    ready(runtime)
    manager, driver = runtime[1:3]
    manager.traffic.acquire("never-drains")
    clock = [0.0]
    budgets = []

    def advance(duration):
        clock[0] += duration

    def pause(replica, *, timeout):
        budgets.append(timeout)
        advance(timeout)

    monkeypatch.setattr(
        "elastic_rollout.runtime.time", NS(monotonic=lambda: clock[0], sleep=advance)
    )
    monkeypatch.setattr(driver, "pause_generation", pause)
    with pytest.raises(GenerationControlTimeout):
        manager.pause_generation()
    assert clock[0] == pytest.approx(150)
    assert budgets[0] == 30 and 0 < budgets[-1] < 30
    assert not manager.traffic.accepting_routes


def test_planned_admission_rejection_waits_for_resume(http_runtime, monkeypatch):
    client, _ = http_runtime
    client.pause_generation()
    acquire = MagicMock(wraps=client.acquire)
    monkeypatch.setattr(client, "acquire", acquire)
    generate = MagicMock(return_value=completion([42], "stop"))
    monkeypatch.setattr("elastic_rollout.generation.request_json", generate)

    async def run():
        waiting, resumed = asyncio.Event(), asyncio.Event()

        async def gate():
            waiting.set()
            await resumed.wait()

        task = asyncio.create_task(
            GenerationClient(client, "policy", wait_for_resume=gate).generate(
                "collision",
                [1],
                {"max_tokens": 4},
            )
        )
        try:
            await asyncio.wait_for(waiting.wait(), 5)
            assert acquire.call_count == 1 and client.status()["inflight"] == 0
            generate.assert_not_called()
            client.resume_generation()
            resumed.set()
            assert (await asyncio.wait_for(task, 5)).token_ids == [42]
            assert acquire.call_count == 2 and generate.call_count == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        ControlError(503, '{"error":"no READY replica"}'),
        ControlError(503, "routing is paused"),
        ControlError(500, '{"code":"ROUTING_PAUSED","route_pause_reasons":["generation"]}'),
        ControlError(503, '{"code":"ROUTING_PAUSED","route_pause_reasons":["shutdown"]}'),
        ControlError(503, '{"code":"ROUTING_PAUSED"}'),
        ConnectionRefusedError("controller down"),
        TimeoutError("response lost"),
    ],
)
def test_unrelated_acquire_failure_never_waits_or_retries(monkeypatch, error):
    client = ControlClient("http://control", "test")
    request = MagicMock(side_effect=error)
    monkeypatch.setattr(client, "_request", request)
    gate = AsyncMock()
    with pytest.raises(type(error)):
        asyncio.run(
            GenerationClient(client, "policy", wait_for_resume=gate).generate(
                "key",
                [1],
                {"max_tokens": 4},
            )
        )
    assert request.call_count == 1
    gate.assert_not_awaited()


def test_immediate_resume_callback_still_backs_off(monkeypatch):
    count = 0

    @asynccontextmanager
    async def route(key):
        nonlocal count
        count += 1
        if count <= 3:
            raise RoutingPaused(503, "planned rejection")
        yield NS(endpoint="http://vllm", serving_policy_version=0)

    sleep = AsyncMock()
    monkeypatch.setattr(
        "elastic_rollout.generation.asyncio", NS(sleep=sleep, to_thread=asyncio.to_thread)
    )
    monkeypatch.setattr(
        "elastic_rollout.generation.request_json", lambda *a: completion([1], "stop")
    )
    asyncio.run(
        GenerationClient(NS(route=route), "policy", wait_for_resume=AsyncMock()).generate(
            "key",
            [1],
            {"max_tokens": 1},
        )
    )
    assert [c.args[0] for c in sleep.call_args_list] == [0.1, 0.2, 0.4]


def test_generation_rpc_timeout_does_not_change_lease_timeouts(http_runtime, monkeypatch):
    client, _ = http_runtime
    lease = client.acquire("record")
    client.release(lease.lease_id)
    request = MagicMock(side_effect=[{}, {}, asdict(lease), asdict(lease), {"released": True}, {}])
    monkeypatch.setattr("elastic_rollout.client.request_json", request)
    client.timeout = 7
    client.pause_generation()
    client.resume_generation()
    client.acquire("record")
    client.renew(lease.lease_id)
    client.release(lease.lease_id)
    client.status()
    assert [call.args[3] for call in request.call_args_list] == [180, 180, 7, 7, 7, 7]


@pytest.mark.parametrize(
    "response", [{}, {"is_paused": "true"}, False, ControlError(404, "missing")]
)
def test_capability_failure_is_explicit_before_physical_control(monkeypatch, response):
    request = MagicMock(
        side_effect=[response] if isinstance(response, Exception) else None, return_value=response
    )
    monkeypatch.setattr("elastic_rollout.engine.vllm.request_json", request)
    with pytest.raises(RuntimeError, match="MVP-1.5B vLLM 0.23 control capability"):
        StandardVLLMControl().pause_generation("http://vllm")
    assert request.call_count == 1 and request.call_args.args[0].endswith("/is_paused")


def test_abort_is_repeated_even_when_already_paused(monkeypatch):
    request = MagicMock(side_effect=[{"is_paused": True}, {}, {"is_paused": True}] * 2)
    monkeypatch.setattr("elastic_rollout.engine.vllm.request_json", request)
    control = StandardVLLMControl()
    control.pause_generation("http://vllm")
    control.pause_generation("http://vllm")
    assert sum(c.args[0].endswith("/pause?mode=abort") for c in request.call_args_list) == 2


@pytest.mark.parametrize("timeout", [0, -1, 151, float("inf"), True])
def test_runtime_budget_cannot_outlive_generation_rpc(timeout):
    with pytest.raises(ValueError):
        DeploymentSpec("test", ReplicaSpec("model"), generation_control_timeout_s=timeout)


def test_pause_rechecks_fresh_processes_after_snapshot_restart(runtime):
    ready(runtime)
    manager, driver = runtime[1:3]
    manager.pause_generation()
    prepare(runtime)
    assert not driver.paused  # Candidate processes started physically running for warmup.
    previous = len(driver.pause_calls)
    status = manager.pause_generation()
    assert status["generation_paused"] and status["inflight"] == 0
    assert len(driver.paused) == 2 and len(driver.pause_calls) == previous + 2


def test_pause_during_startup_rechecks_new_ready_targets(runtime):
    controller, manager, driver, _ = runtime
    controller.reconcile()  # Processes started, but not READY yet.
    manager.pause_generation()
    assert not driver.pause_calls
    controller.reconcile()
    manager.pause_generation()
    assert len(driver.paused) == 2


def test_late_resume_cannot_undo_another_callers_fail_closed_timeout(runtime, monkeypatch):
    ready(runtime)
    manager, driver = runtime[1:3]
    manager.pause_generation()
    entered, finish = threading.Event(), threading.Event()
    original = driver.resume_generation

    def resume(replica, **kwargs):
        entered.set()
        assert finish.wait(5)
        original(replica, **kwargs)

    monkeypatch.setattr(driver, "resume_generation", resume)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(manager.resume_generation)
        try:
            assert entered.wait(5)
            # Simulate the lock-wait timeout path while another caller is in I/O.
            manager._generation_timeout("pause")
        finally:
            finish.set()
        with pytest.raises(GenerationControlTimeout):
            pending.result(5)
    assert not manager.traffic.accepting_routes
    assert manager._generation_control_failed
    manager.resume_generation()  # Explicit recovery can clear its own gate.
    assert manager.traffic.accepting_routes


def test_physical_state_must_match_control_response(monkeypatch):
    request = MagicMock(side_effect=[{"is_paused": False}, {}, {"is_paused": False}])
    monkeypatch.setattr("elastic_rollout.engine.vllm.request_json", request)
    with pytest.raises(RuntimeError, match="did not confirm is_paused=True"):
        StandardVLLMControl().pause_generation("http://vllm")
