"""Linux vLLM process-group guardian: parent Ray actor pipe EOF triggers cleanup.

Ray actor termination does not run Python destructors. Keeping the guardian's
stdin pipe open only in its owner makes actor death observable without Ray APIs.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)


def _wait_group_exit(process, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        process.poll()  # Reap the group leader before testing group existence.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _cleanup(process) -> None:
    logger.info("stopping vLLM process group pgid=%s with SIGTERM", process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not _wait_group_exit(process, 10):
        logger.warning("vLLM process group pgid=%s requires SIGKILL", process.pid)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not _wait_group_exit(process, 5):
            raise RuntimeError(f"vLLM process group {process.pid} survived cleanup")
    process.wait(timeout=1)
    logger.info("vLLM process group pgid=%s cleanup verified", process.pid)


def main() -> int:
    if os.name != "posix":
        raise RuntimeError("Ascend vLLM process supervision requires Linux")
    logging.basicConfig(level=logging.INFO)
    shutdown = threading.Event()
    # Protect the entire spawn/cleanup interval, including signals during Popen.
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    acknowledgement_fd = os.environ.pop("ELASTIC_ROLLOUT_CLEANUP_FD", None)
    process = None

    def watch_parent():
        # Buffered stdin can hold a Python I/O lock across shutdown when SIGTERM
        # arrives before EOF. A daemon blocked in raw read has no such lock.
        try:
            while os.read(sys.stdin.fileno(), 4096):
                pass
        finally:
            shutdown.set()

    try:
        threading.Thread(target=watch_parent, daemon=True).start()
        process = subprocess.Popen(sys.argv[1:], start_new_session=True, stdin=subprocess.DEVNULL)
        logger.info("supervisor pid=%s started vLLM pid/pgid=%s", os.getpid(), process.pid)
        while process.poll() is None and not shutdown.wait(0.1):
            pass
    finally:
        if process is not None:
            _cleanup(process)
            # A separate pipe proves cleanup ran; process exit alone proves nothing.
            # Popen closes this descriptor in vLLM, so a killed guardian yields EOF.
            if acknowledgement_fd is not None:
                try:
                    os.write(int(acknowledgement_fd), b"CLEAN\n")
                except BrokenPipeError:
                    pass  # Owner died; the process group has still been cleaned.
    return process.returncode if process.returncode >= 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
