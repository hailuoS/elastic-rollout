"""Small HTTP control client; prompt/token data never goes to this service."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .api import DEFAULT_POLICY_UPDATE_TIMEOUT_S, DeploymentSpec, RouteLease

logger = logging.getLogger(__name__)


class ControlError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def request_json(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 10,
    token: str | None = None,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except HTTPError as exc:
        raise ControlError(exc.code, exc.read().decode(errors="replace")) from exc


async def complete_on_cancel(awaitable):
    """Finish a bounded I/O call before propagating cancellation.

    A cancelled to_thread call keeps running. In particular, an acquire may have
    succeeded remotely even when the caller was cancelled before reading its ID.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


class ControlClient:
    def __init__(
        self, endpoint: str, deployment_id: str, token: str | None = None, timeout: float = 10
    ):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("control endpoint must include http:// or https://")
        self.endpoint = endpoint.rstrip("/")
        self.deployment_id, self.token, self.timeout = deployment_id, token, timeout

    @property
    def path(self) -> str:
        return f"/v1/deployments/{quote(self.deployment_id, safe='')}"

    def _request(self, path: str, method: str = "GET", payload: dict | None = None) -> dict:
        return request_json(self.endpoint + path, method, payload, self.timeout, self.token)

    def ensure(self, spec: DeploymentSpec) -> dict:
        if spec.deployment_id != self.deployment_id:
            raise ValueError("deployment_id does not match client")
        return self._request(self.path, "PUT", asdict(spec))

    def status(self) -> dict:
        return self._request(self.path)

    def replicas(self) -> list[dict]:
        return self._request(self.path + "/replicas")["replicas"]

    def set_desired_replicas(self, count: int) -> dict:
        return self._request(self.path + "/desired", "PUT", {"desired_replicas": count})

    def begin_policy_update(self, snapshot_path: str) -> dict:
        return self._request(
            self.path + "/policy-updates", "POST", {"snapshot_path": snapshot_path}
        )

    def get_policy_update(self, update_id: str) -> dict:
        return self._request(self.path + f"/policy-updates/{quote(update_id, safe='')}")

    def commit_policy_update(self, update_id: str, policy_version: int) -> dict:
        return self._request(
            self.path + f"/policy-updates/{quote(update_id, safe='')}/commit",
            "POST",
            {"policy_version": policy_version},
        )

    def wait_policy_update(
        self,
        update_id: str,
        desired_state: str = "PREPARED",
        timeout: float = DEFAULT_POLICY_UPDATE_TIMEOUT_S,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            update = self.get_policy_update(update_id)
            if update["state"] == "FAILED":
                raise RuntimeError(f"policy update {update_id} FAILED: {update.get('error')}")
            if update["state"] == desired_state:
                return update
            if update["state"] == "COMMITTED":
                raise RuntimeError(f"policy update {update_id} already COMMITTED")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"policy update {update_id} did not reach {desired_state}")
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))

    def acquire(self, routing_key: str) -> RouteLease:
        return RouteLease(
            **self._request(self.path + "/routes", "POST", {"routing_key": routing_key})
        )

    def release(self, lease_id: str) -> bool:
        return self._request(self.path + f"/leases/{quote(lease_id, safe='')}", "DELETE")[
            "released"
        ]

    def renew(self, lease_id: str) -> RouteLease:
        return RouteLease(
            **self._request(self.path + f"/leases/{quote(lease_id, safe='')}", "PUT", {})
        )

    def wait_ready(self, count: int, timeout: float = 900) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            if status["ready_replicas"] == count and status["total_replicas"] == count:
                return status
            time.sleep(0.2)
        raise TimeoutError(f"deployment did not converge to {count}: {self.status()}")

    @asynccontextmanager
    async def route(self, routing_key: str):
        acquire_task = asyncio.create_task(asyncio.to_thread(self.acquire, routing_key))
        try:
            lease = await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            # Recover a successful but cancelled acquire and release its lease.
            try:
                lease = await acquire_task
                await asyncio.to_thread(self.release, lease.lease_id)
            finally:
                raise

        async def heartbeat():
            current = lease
            while True:
                delay = max(0.05, (current.expires_at - time.time()) / 3)
                await asyncio.sleep(delay)
                try:
                    current = await asyncio.to_thread(self.renew, current.lease_id)
                except Exception:
                    # Failures leave TTL + backend idle check as the safety net.
                    logger.exception("route lease renewal failed")
                    return

        task = asyncio.create_task(heartbeat())
        try:
            yield lease
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            try:
                await complete_on_cancel(asyncio.to_thread(self.release, lease.lease_id))
            except Exception:
                logger.exception("route release failed; runtime will expire its TTL")
