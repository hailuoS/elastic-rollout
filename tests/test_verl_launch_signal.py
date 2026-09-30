"""Real SIGTERM delivery after a real Ray.init, on an existing Linux Ray cluster."""

import json
import os
import select
import signal
import subprocess
import sys
import textwrap
import time

import pytest


@pytest.mark.cluster
@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM process semantics required")
def test_serve_sigterm_after_real_ray_init_closes_controller(tmp_path):
    address = os.environ.get("ELASTIC_ROLLOUT_TEST_RAY_ADDRESS")
    if not address:
        pytest.skip("set ELASTIC_ROLLOUT_TEST_RAY_ADDRESS to an existing Ray cluster")
    pytest.importorskip("ray")
    config = tmp_path / "empty-core.json"
    config.write_text(json.dumps({"slots": [], "deployments": []}), encoding="utf-8")
    closed = tmp_path / "closed"
    script = textwrap.dedent(
        """
        import sys
        import threading
        from pathlib import Path
        from elastic_rollout import cli
        config, closed, address = sys.argv[1:]

        class Controller:
            def __init__(self, *_):
                pass

            def close(self, **_):
                Path(closed).write_text("closed", encoding="utf-8")

        class Server:
            def __init__(self):
                self.stop = threading.Event()

            def serve_forever(self, **_):
                print("READY", flush=True)
                self.stop.wait()

            def shutdown(self):
                self.stop.set()

            def server_close(self):
                pass

        server = Server()
        cli.Controller = Controller
        cli.make_server = lambda *_: server
        cli.Reconciler = lambda *_: type("Reconciler", (), {"start": lambda self: None,
                                                    "stop": lambda self: None})()
        cli.VLLMServiceDriver = lambda *_: object()
        sys.argv = ["elastic-rollout", "serve", "--config", config,
                    "--ray-address", address]
        cli.main()
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(config), str(closed), address],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    try:
        deadline = time.monotonic() + 60
        output = b""
        while time.monotonic() < deadline:
            readable, _, _ = select.select([proc.stdout], [], [], 1)
            if readable:
                output += os.read(proc.stdout.fileno(), 4096)
                if b"READY\n" in output:
                    break
            if proc.poll() is not None:
                pytest.fail(
                    "serve exited before Ray initialization: " + output.decode(errors="replace")
                )
        else:
            pytest.fail("serve did not finish Ray initialization within 60 seconds")
        proc.send_signal(signal.SIGTERM)
        remaining, _ = proc.communicate(timeout=30)
        assert proc.returncode == 0, (output + remaining).decode(errors="replace")
        assert closed.read_text(encoding="utf-8") == "closed"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
