import asyncio
import threading
import time
from dataclasses import replace

import pytest

from elastic_rollout.client import ControlClient
from elastic_rollout.server import make_server
from tests.helpers import ready


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
