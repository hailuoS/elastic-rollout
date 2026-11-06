"""Async generation I/O shared by framework adapters, outside the core runtime."""

import asyncio
import json

from elastic_rollout.client import ControlError, complete_on_cancel
from elastic_rollout.generation import GenerationClient


class AsyncGenerationClient(GenerationClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, request=self._request, **kwargs)
        self._sessions = {}

    async def generate(self, *args, **kwargs):
        import aiohttp

        # A client may be reused across workflow event loops. Share a pool only
        # within one loop and keep it alive through overlapping logical requests.
        loop = asyncio.get_running_loop()
        if loop not in self._sessions:
            self._sessions[loop] = [
                aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=0),
                    cookie_jar=aiohttp.DummyCookieJar(),
                ),
                0,
            ]
        state = self._sessions[loop]
        state[1] += 1
        try:
            return await super().generate(*args, **kwargs)
        finally:
            state[1] -= 1
            if state[1] == 0:
                del self._sessions[loop]
                await complete_on_cancel(state[0].close())

    async def _request(self, url, method, payload, timeout):
        import aiohttp

        session = self._sessions[asyncio.get_running_loop()][0]
        async with session.request(
            method,
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout, connect=30),
            allow_redirects=False,
        ) as response:
            body = await response.read()
            # Generation is not idempotent. Do not follow redirects or retry.
            if response.status >= 300:
                raise ControlError(response.status, body.decode(errors="replace"))
            return json.loads(body) if body else {}
