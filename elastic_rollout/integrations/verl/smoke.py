"""Optional NPU smoke assertions; the normal launcher never imports this module."""

import asyncio
import logging
import socket
import time

from elastic_rollout.generation import GenerationClient

logger = logging.getLogger(__name__)


def probe_generation(client, spec, core, stop, prompt_ids: list[int]) -> dict:
    if any(type(token) is not int or token < 0 for token in prompt_ids):
        raise ValueError("smoke prompt IDs must be nonnegative integers")
    deadline = time.monotonic() + spec.startup_timeout_s
    while time.monotonic() < deadline:
        if stop[0] or core.poll() is not None:
            raise InterruptedError("Core exited or launch interrupted before smoke generation")
        status = client.status()
        version = status.get("serving_policy_version")
        if (
            version is not None
            and version > spec.initial_version
            and status.get("serving_ready_for_external_update")
        ):
            break
        time.sleep(0.2)
    else:
        raise TimeoutError(
            "trainer finished without a stable committed policy newer than bootstrap"
        )
    generator = GenerationClient(
        client, status["spec"]["replica"]["served_model_name"], request_timeout=60
    )
    completion = asyncio.run(
        generator.generate(
            "mvp2b-post-update-smoke", prompt_ids, {"max_tokens": 4, "temperature": 0}
        )
    )
    if not completion.token_ids or any(item != version for item in completion.versions):
        raise RuntimeError("post-update generation did not use the committed serving version")
    return {
        "initial_version": spec.initial_version,
        "final_version": version,
        "post_update_token_versions": completion.versions,
        "generated_tokens": len(completion.token_ids),
    }


def rollout_ports_closed(slots: list[dict], timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        open_slots = []
        for slot in slots:
            try:
                with socket.create_connection((slot["host"], slot["port"]), timeout=0.5):
                    open_slots.append(slot["slot_id"])
            except OSError:
                pass
        if not open_slots:
            return True
        if time.monotonic() >= deadline:
            logger.error("rollout serving ports remain open after Core close: %s", open_slots)
            return False
        time.sleep(0.2)
