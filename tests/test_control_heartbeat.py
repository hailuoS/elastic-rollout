import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError

import pytest

from elastic_rollout.client import ControlError
from elastic_rollout.generation import GenerationClient


async def wait_until(predicate, timeout=3):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


def short_lease_ttl(runtime, ttl=0.6):
    manager = runtime[1]
    manager.spec = replace(manager.spec, lease_ttl=ttl)
    manager.traffic.spec = manager.spec
    return manager


@pytest.fixture
def blocked_generation_server():
    release = threading.Event()
    both_started = threading.Event()
    lock = threading.Lock()
    started = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal started
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                started += 1
                if started == 2:
                    both_started.set()
            # Keep both default-executor threads inside real generation HTTP calls.
            release.wait(timeout=10)
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "logprobs": {
                                "tokens": ["token_id:42"],
                                "token_logprobs": [-0.1],
                            },
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
        yield f"http://127.0.0.1:{server.server_port}", both_started, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_heartbeat_survives_default_executor_starvation(http_runtime, blocked_generation_server):
    client, runtime = http_runtime
    manager = short_lease_ttl(runtime)
    endpoint, both_started, release = blocked_generation_server
    for replica in manager.traffic.replicas.values():
        replica.endpoint = endpoint
    generator = GenerationClient(client, "policy")

    async def run():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        tasks = [
            asyncio.create_task(generator.generate(str(i), [i], {"max_tokens": 1})) for i in (1, 2)
        ]
        try:
            await wait_until(both_started.is_set)
            with manager.traffic.lock:
                initial = dict(manager.traffic.leases)
            assert len(initial) == 2
            # Do not unblock generation until the original leases would have expired.
            await asyncio.sleep(max(r.expires_at for r in initial.values()) - time.time() + 0.1)
            manager.traffic.expire()
            with manager.traffic.lock:
                live = dict(manager.traffic.leases)
            assert set(live) == set(initial)
            assert all(live[k].expires_at > initial[k].expires_at for k in initial)

            # Acquire and release must also work while default workers remain occupied.
            async def probe():
                async with client.route("control-only-probe"):
                    pass

            await asyncio.wait_for(probe(), timeout=2)
            assert client.status()["inflight"] == 2
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert all(task.result().token_ids == [42] for task in tasks)

    asyncio.run(run())
    assert client.status()["inflight"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("temporary timeout"),
        ConnectionResetError("connection reset"),
        URLError("temporary connection failure"),
        ControlError(500, "temporary controller failure"),
        ControlError(503, "temporarily unavailable"),
    ],
)
def test_heartbeat_retries_transient_failure(http_runtime, monkeypatch, caplog, failure):
    client, runtime = http_runtime
    manager = short_lease_ttl(runtime)
    original_renew = client.renew
    original_acquire = client.acquire
    renewed = threading.Event()
    attempts = []
    acquisitions = []
    caplog.set_level(logging.DEBUG, logger="elastic_rollout.client")

    def acquire(key):
        acquisitions.append(key)
        return original_acquire(key)

    def flaky_renew(lease_id, **kwargs):
        attempts.append(lease_id)
        if len(attempts) == 1:
            raise failure
        result = original_renew(lease_id, **kwargs)
        renewed.set()
        return result

    monkeypatch.setattr(client, "renew", flaky_renew)
    monkeypatch.setattr(client, "acquire", acquire)

    async def run():
        async with client.route("retry-renew") as lease:
            await wait_until(renewed.is_set)
            assert len(attempts) >= 2
            assert set(attempts) == {lease.lease_id}
            manager.traffic.expire()
            assert manager.traffic.leases[lease.lease_id].expires_at > lease.expires_at
            assert client.status()["inflight"] == 1

    asyncio.run(run())
    assert acquisitions == ["retry-renew"]
    assert client.status()["inflight"] == 0
    for field in (
        "lease_id=",
        "submitted_at=",
        "started_at=",
        "ended_at=",
        "expires_at=",
        "latency_s=",
        "queue_wait_s=",
        "remaining_ttl_s=",
    ):
        assert field in caplog.text
    assert "retry-renew" not in caplog.text  # No request payload/routing key logging.


def test_heartbeat_stops_on_missing_lease_without_reacquiring(http_runtime, monkeypatch):
    client, runtime = http_runtime
    short_lease_ttl(runtime)
    lease = client.acquire("deleted-lease")
    # The actual control HTTP server will respond with 404 to renewal.
    assert client.release(lease.lease_id)
    original_renew = client.renew
    attempts = []

    def renew(lease_id, **kwargs):
        attempts.append(lease_id)
        return original_renew(lease_id, **kwargs)

    def no_acquire(*args):
        pytest.fail("heartbeat must never acquire another lease")

    monkeypatch.setattr(client, "renew", renew)
    monkeypatch.setattr(client, "acquire", no_acquire)

    async def run():
        await asyncio.wait_for(client._heartbeat(lease), timeout=2)

    asyncio.run(run())
    assert attempts == [lease.lease_id]
    assert client.status()["inflight"] == 0


def test_heartbeat_retries_stop_at_known_expiry(http_runtime, monkeypatch):
    client, runtime = http_runtime
    short_lease_ttl(runtime)
    lease = client.acquire("always-times-out")
    attempts = []

    def renew(lease_id, *, timeout=None):
        attempts.append((lease_id, time.time(), timeout))
        raise TimeoutError("still unavailable")

    monkeypatch.setattr(client, "renew", renew)

    async def run():
        await asyncio.wait_for(client._heartbeat(lease), timeout=2)

    asyncio.run(run())
    assert len(attempts) >= 2
    assert all(lease_id == lease.lease_id for lease_id, _, _ in attempts)
    assert all(
        start < lease.expires_at and 0 < timeout <= client.timeout for _, start, timeout in attempts
    )
    assert time.time() >= lease.expires_at
    assert client.status()["inflight"] == 0


def test_cancelled_acquire_recovers_and_releases_lease(http_runtime, monkeypatch):
    client, _ = http_runtime
    original_acquire = client.acquire
    acquired = threading.Event()
    finish = threading.Event()

    def acquire(key):
        lease = original_acquire(key)
        acquired.set()
        assert finish.wait(timeout=3)
        return lease

    monkeypatch.setattr(client, "acquire", acquire)

    async def run():
        async def worker():
            async with client.route("cancel-during-acquire"):
                pytest.fail("cancelled acquire must not enter route body")

        task = asyncio.create_task(worker())
        try:
            await wait_until(acquired.is_set)
            task.cancel()
            await asyncio.sleep(0)
        finally:
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert client.status()["inflight"] == 0


def test_cancelled_route_finishes_renew_before_release(http_runtime, monkeypatch):
    client, runtime = http_runtime
    short_lease_ttl(runtime, ttl=2)
    original_renew, original_release = client.renew, client.release
    renewing = threading.Event()
    finish = threading.Event()
    completed = []

    def renew(lease_id, **kwargs):
        renewing.set()
        assert finish.wait(timeout=3)
        result = original_renew(lease_id, **kwargs)
        completed.append("renew")
        return result

    def release(lease_id):
        completed.append("release")
        return original_release(lease_id)

    monkeypatch.setattr(client, "renew", renew)
    monkeypatch.setattr(client, "release", release)

    async def run():
        async def worker():
            async with client.route("cancel-during-renew"):
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        try:
            await wait_until(renewing.is_set)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done()
            assert completed == []
        finally:
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert completed == ["renew", "release"]
    assert client.status()["inflight"] == 0
