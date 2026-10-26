"""Shared standard-library HTTP transport for serving and control clients."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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
