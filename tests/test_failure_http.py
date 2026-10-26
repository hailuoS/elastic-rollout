import asyncio
import logging

from tests.http_fixtures import short_lease_ttl


def test_crash_lease_http_404_stops_heartbeat_once(http_runtime, monkeypatch, caplog):
    client, runtime = http_runtime
    controller, manager, driver, _ = runtime
    short_lease_ttl(runtime)
    lease = client.acquire("crashing-replica")
    other = client.acquire("healthy-replica")
    driver.processes.remove(lease.replica_id)
    controller.reconcile()
    renew = client.renew
    attempts = []

    def counted_renew(lease_id, **kwargs):
        attempts.append(lease_id)
        return renew(lease_id, **kwargs)

    monkeypatch.setattr(client, "renew", counted_renew)
    caplog.set_level(logging.DEBUG, logger="elastic_rollout.client")

    async def run():
        await asyncio.wait_for(client._heartbeat(lease), 2)
        await asyncio.sleep(0.15)

    asyncio.run(run())
    assert attempts == [lease.lease_id]
    assert not client.release(lease.lease_id)
    assert client.renew(other.lease_id).replica_id == other.replica_id
    assert client.status()["inflight"] == 1
    assert client.release(other.lease_id)
    assert client.status()["inflight"] == 0
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is None
    assert "status=404" in warnings[0].getMessage()
    assert lease.lease_id in warnings[0].getMessage()
    assert lease.replica_id in warnings[0].getMessage()
    assert "lease renew retry" not in caplog.text
