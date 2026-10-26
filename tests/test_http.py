import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from elastic_rollout.client import ControlError, request_json
from elastic_rollout.runtime import Reconciler


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
