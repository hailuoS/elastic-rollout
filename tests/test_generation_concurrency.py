import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from elastic_rollout.generation import GenerationClient
from tests.http_fixtures import short_lease_ttl, wait_until


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
