"""Linux vLLM process-group guardian: parent Ray actor pipe EOF triggers cleanup.

Ray actor termination does not run Python destructors. Keeping the guardian's
stdin pipe open only in its owner makes actor death observable without Ray APIs.
"""

import os
import signal
import subprocess
import sys
import threading
import time


def main() -> int:
    if os.name != "posix":
        raise RuntimeError("Ascend vLLM process supervision requires Linux")
    process = subprocess.Popen(sys.argv[1:], start_new_session=True, stdin=subprocess.DEVNULL)
    shutdown = threading.Event()

    def watch_parent():
        sys.stdin.buffer.read()
        shutdown.set()

    threading.Thread(target=watch_parent, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    try:
        while process.poll() is None and not shutdown.wait(0.1):
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            process.poll()  # Reap the group leader before testing group existence.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    return process.returncode if process.returncode >= 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
