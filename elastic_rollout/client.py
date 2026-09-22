"""Small HTTP control client; prompt/token data never goes to this service."""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import copy_context
from dataclasses import asdict
from functools import partial
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .api import (
    DEFAULT_POLICY_UPDATE_TIMEOUT_S,
    GENERATION_CONTROL_RPC_TIMEOUT_S,
    DeploymentSpec,
    RouteLease,
)

logger = logging.getLogger(__name__)

# Shared by route clients in this process, not by generation HTTP calls. Threads
# start lazily and ThreadPoolExecutor joins them at interpreter shutdown.
_CONTROL_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="elastic-control")


async def _control_io(function, *args):
    # Preserve the context propagation provided by asyncio.to_thread().
    call = partial(copy_context().run, function, *args)
    return await asyncio.get_running_loop().run_in_executor(_CONTROL_EXECUTOR, call)


class ControlError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class RoutingPaused(ControlError):
    """An explicit, rejected AcquireRoute during a planned pause; no lease was created."""


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

    A cancelled executor call keeps running. In particular, an acquire may have
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

    def _request(
        self,
        path: str,
        method: str = "GET",
        payload: dict | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        return request_json(
            self.endpoint + path,
            method,
            payload,
            self.timeout if timeout is None else timeout,
            self.token,
        )

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

    def pause_generation(self) -> dict:
        return self._request(
            self.path + "/generation/pause", "POST", {}, timeout=GENERATION_CONTROL_RPC_TIMEOUT_S
        )

    def resume_generation(self) -> dict:
        return self._request(
            self.path + "/generation/resume", "POST", {}, timeout=GENERATION_CONTROL_RPC_TIMEOUT_S
        )

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
        try:
            result = self._request(self.path + "/routes", "POST", {"routing_key": routing_key})
        except ControlError as exc:
            # Only a structured rejection proves no lease was created. Never
            # infer this from an arbitrary 503 or retry an ambiguous network error.
            try:
                detail = json.loads(str(exc))
            except (ValueError, TypeError):
                detail = None
            if isinstance(detail, dict) and exc.status == 503:
                reasons = detail.get("route_pause_reasons")
                if (
                    detail.get("code") == "ROUTING_PAUSED"
                    and isinstance(reasons, list)
                    and reasons
                    and all(r in ("generation", "policy_update") for r in reasons)
                ):
                    raise RoutingPaused(exc.status, str(exc)) from exc
            raise
        return RouteLease(**result)

    def release(self, lease_id: str) -> bool:
        return self._request(self.path + f"/leases/{quote(lease_id, safe='')}", "DELETE")[
            "released"
        ]

    def renew(self, lease_id: str, *, timeout: float | None = None) -> RouteLease:
        return RouteLease(
            **self._request(
                self.path + f"/leases/{quote(lease_id, safe='')}",
                "PUT",
                {},
                timeout=timeout,
            )
        )

    def wait_ready(self, count: int, timeout: float = 900) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            if status["ready_replicas"] == count and status["total_replicas"] == count:
                return status
            time.sleep(0.2)
        raise TimeoutError(f"deployment did not converge to {count}: {self.status()}")

    def _renew_once(self, lease: RouteLease, submitted: float) -> RouteLease:
        started_at, started = time.time(), time.monotonic()
        remaining = lease.expires_at - started_at
        logger.debug(
            "lease renew start lease_id=%s started_at=%.6f queue_wait_s=%.6f "
            "expires_at=%.6f remaining_ttl_s=%.6f",
            lease.lease_id,
            started_at,
            started - submitted,
            lease.expires_at,
            remaining,
        )
        current, outcome = lease, "error"
        try:
            # Recheck after executor queuing; never send a retry after known expiry.
            if remaining <= 0:
                raise TimeoutError("lease expired before renewal could start")
            current = self.renew(lease.lease_id, timeout=min(self.timeout, remaining))
            outcome = "ok"
            return current
        finally:
            ended_at = time.time()
            logger.debug(
                "lease renew end lease_id=%s ended_at=%.6f latency_s=%.6f "
                "expires_at=%.6f remaining_ttl_s=%.6f outcome=%s",
                lease.lease_id,
                ended_at,
                time.monotonic() - started,
                current.expires_at,
                current.expires_at - ended_at,
                outcome,
            )

    async def _heartbeat(self, lease: RouteLease) -> None:
        current = lease
        backoff = 0.1
        delay = max(0.05, (current.expires_at - time.time()) / 3)
        attempt = 0
        while True:
            await asyncio.sleep(delay)
            submitted_at, submitted = time.time(), time.monotonic()
            remaining = current.expires_at - submitted_at
            if remaining <= 0:
                logger.error(
                    "lease heartbeat stopped: known TTL expired lease_id=%s expires_at=%.6f",
                    current.lease_id,
                    current.expires_at,
                )
                return
            attempt += 1
            logger.debug(
                "lease renew submit lease_id=%s submitted_at=%.6f attempt=%s "
                "expires_at=%.6f remaining_ttl_s=%.6f",
                current.lease_id,
                submitted_at,
                attempt,
                current.expires_at,
                remaining,
            )
            try:
                # Finish any running renewal before route() releases the lease on cancel.
                current = await complete_on_cancel(
                    _control_io(self._renew_once, current, submitted)
                )
            except Exception as exc:
                remaining = current.expires_at - time.time()
                if isinstance(exc, ControlError) and exc.status in (404, 410):
                    # A missing lease may have expired, been released or been
                    # invalidated by a failed replica. Do not invent its cause.
                    logger.warning(
                        "route lease unavailable; heartbeat stopped "
                        "lease_id=%s replica_id=%s status=%s",
                        current.lease_id,
                        current.replica_id,
                        exc.status,
                    )
                    logger.debug("terminal lease renewal response", exc_info=True)
                    return
                retryable = (
                    500 <= exc.status < 600
                    if isinstance(exc, ControlError)
                    else isinstance(exc, (OSError, URLError, HTTPException))
                )
                if not retryable or remaining <= 0:
                    logger.exception(
                        "lease heartbeat stopped lease_id=%s expires_at=%.6f "
                        "remaining_ttl_s=%.6f retryable=%s",
                        current.lease_id,
                        current.expires_at,
                        remaining,
                        retryable,
                    )
                    return
                # Only retry renewal of this lease, bounded by its last known expiry.
                delay = min(backoff, remaining)
                backoff = min(backoff * 2, 1.0)
                logger.warning(
                    "lease renew retry lease_id=%s attempt=%s error=%s status=%s "
                    "expires_at=%.6f remaining_ttl_s=%.6f retry_in_s=%.6f",
                    current.lease_id,
                    attempt,
                    type(exc).__name__,
                    getattr(exc, "status", None),
                    current.expires_at,
                    remaining,
                    delay,
                )
            else:
                backoff = 0.1
                delay = max(0.05, (current.expires_at - time.time()) / 3)

    @asynccontextmanager
    async def route(self, routing_key: str):
        acquire_task = asyncio.create_task(_control_io(self.acquire, routing_key))
        try:
            lease = await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            # Recover a successful but cancelled acquire and release its lease.
            try:
                lease = await acquire_task
                await complete_on_cancel(_control_io(self.release, lease.lease_id))
            finally:
                raise

        task = asyncio.create_task(self._heartbeat(lease))
        try:
            yield lease
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            try:
                await complete_on_cancel(_control_io(self.release, lease.lease_id))
            except Exception:
                logger.exception("route release failed; runtime will expire its TTL")
