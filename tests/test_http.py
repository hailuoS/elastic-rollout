import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_runtime import ready

from elastic_rollout.client import ControlClient, ControlError, request_json
from elastic_rollout.generation import GenerationClient, parse_completion
from elastic_rollout.runtime import Reconciler
from elastic_rollout.server import make_server


@pytest.fixture
def http_runtime(runtime):
    controller, manager, driver, clock = runtime
    # Real-time TTL is needed for client heartbeat tests.
    manager.traffic.clock = time.time
    ready(runtime)
    server = make_server(controller, port=0, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(f"http://127.0.0.1:{server.server_port}", "test", "test-token")
    try:
        yield client, runtime
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_control_api_and_auth(http_runtime):
    client, runtime = http_runtime
    assert client.status()["ready_replicas"] == 2
    lease = client.acquire("conversation")
    assert client.renew(lease.lease_id).lease_id == lease.lease_id
    assert client.release(lease.lease_id)
    assert not client.release(lease.lease_id)
    assert len(client.replicas()) == 2
    assert client.set_desired_replicas(4)["desired_replicas"] == 4
    with pytest.raises(ControlError) as error:
        request_json(client.endpoint + "/health")
    assert error.value.status == 401
    with pytest.raises(ControlError) as error:
        request_json(client.endpoint + "/elastic/generate", "POST", {}, token=client.token)
    assert error.value.status == 404


def test_close_closes_admission_before_waiting_for_generation_lock(http_runtime, monkeypatch):
    client, (controller, manager, _, _) = http_runtime
    manager.spec = replace(manager.spec, drain_grace_period=0)
    lease = client.acquire("existing")
    gate_closed = threading.Event()
    pause_routes = manager.traffic.pause_routes

    def observe_pause(reason):
        pause_routes(reason)
        if reason == "shutdown":
            gate_closed.set()

    monkeypatch.setattr(manager.traffic, "pause_routes", observe_pause)
    manager.generation_lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(controller.close, timeout=5)
        try:
            assert gate_closed.wait(5)
            assert not closing.done()
            with pytest.raises(ControlError) as error:
                client.acquire("new")
            assert error.value.status == 503
            assert client.renew(lease.lease_id).lease_id == lease.lease_id
            assert client.release(lease.lease_id)
        finally:
            manager.generation_lock.release()
        assert closing.result(timeout=5) is None
    assert controller.resources.allocated == 0


def test_concurrent_http_route_acquire_and_release(http_runtime):
    client, _ = http_runtime
    count = 128
    # Synchronize each burst so requests compete at the HTTP accept queue.
    barrier = threading.Barrier(count, timeout=10)

    def acquire(index):
        barrier.wait()
        return client.acquire(f"concurrent-route-{index}")

    def release(lease):
        barrier.wait()
        return client.release(lease.lease_id)

    assert client.status()["inflight"] == 0
    with ThreadPoolExecutor(max_workers=count) as pool:
        # Consume every result: connection resets and other HTTP errors fail the test.
        leases = list(pool.map(acquire, range(count)))
        assert len({lease.lease_id for lease in leases}) == count
        assert client.status()["inflight"] == count
        released = list(pool.map(release, leases))

    assert all(released)
    assert client.status()["inflight"] == 0


def test_replicas_api_retains_assignment_after_failure_and_slot_reuse(http_runtime):
    client, runtime = http_runtime
    controller, manager, driver, _ = runtime
    failed = next(iter(manager.traffic.replicas.values()))
    allocation = failed.resource.allocation
    assigned = tuple(map(str, range(allocation.num_devices)))
    allocation.actual_device_ids = assigned
    driver.processes.remove(failed.replica_id)
    controller.reconcile()  # Clean failed lease and allocate its slot to a new replica.
    record = next(r for r in client.replicas() if r["replica_id"] == failed.replica_id)
    assert record["state"] == "FAILED" and record["resource"] is None
    assert record["allocation"] == {
        "slot_id": allocation.slot_id,
        "num_devices": allocation.num_devices,
        "actual_device_ids": list(assigned),
    }
    replacement = next(
        r
        for r in manager.traffic.replicas.values()
        if r.resource is not None and r.resource.slot.slot_id == allocation.slot_id
    )
    assert replacement.allocation is not allocation
    assert replacement.allocation.actual_device_ids == ()


@pytest.fixture
def inference_server():
    requests = []
    aborted = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal aborted
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            # First call exercises full-context resume; second finishes.
            reason = "stop" if aborted else "abort"
            aborted = True
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": reason,
                            "logprobs": {"tokens": ["token_id:42"], "token_logprobs": [-0.1]},
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_direct_generation_resume_versions_and_release(http_runtime, inference_server):
    client, runtime = http_runtime
    endpoint, requests = inference_server
    manager = runtime[1]
    for replica in manager.traffic.replicas.values():
        replica.endpoint = endpoint
        replica.serving_policy_version = 7
    generator = GenerationClient(client, "policy")
    params = {"max_tokens": 3, "temperature": 0}
    result = asyncio.run(generator.generate("trajectory", [1, 2], params))
    assert result.token_ids == [42, 42]
    assert result.versions == [7, 7]
    assert requests[1]["prompt"] == [1, 2, 42]
    assert requests[1]["max_tokens"] == 2
    assert params["max_tokens"] == 3
    assert manager.status()["inflight"] == 0


def test_release_on_exception_and_cancel(http_runtime):
    client, runtime = http_runtime

    async def run():
        with pytest.raises(ValueError):
            async with client.route("error"):
                raise ValueError("generation failed")
        entered = asyncio.Event()

        async def worker():
            async with client.route("cancel"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert runtime[1].status()["inflight"] == 0


def test_background_reconciler_converges(http_runtime):
    client, runtime = http_runtime
    reconciler = Reconciler(runtime[0], interval=0.01)
    reconciler.start()
    try:
        client.set_desired_replicas(4)
        assert client.wait_ready(4, timeout=2)["ready_replicas"] == 4
    finally:
        reconciler.stop()


@pytest.mark.parametrize(
    "tokens,probs", [(["text"], [-1.0]), (["token:1"], []), (["token:1"], [None])]
)
def test_malformed_token_outputs_rejected(tokens, probs):
    with pytest.raises(ValueError):
        parse_completion(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "logprobs": {"tokens": tokens, "token_logprobs": probs},
                    }
                ]
            }
        )
