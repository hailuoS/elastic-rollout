import os
import select
import signal
import subprocess
import sys
import time

import pytest

from elastic_rollout.integrations.areal.lifecycle import graceful_sigterm


def test_sigterm_unwinds_cleanup_and_restores_handler():
    previous = signal.getsignal(signal.SIGTERM)
    closed = []
    with pytest.raises(SystemExit) as exc:
        with graceful_sigterm():
            handler = signal.getsignal(signal.SIGTERM)
            try:
                handler(signal.SIGTERM, None)
            finally:
                assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
                closed.append(True)
    assert exc.value.code == 143
    assert closed == [True]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_exception_also_restores_handler():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError):
        with graceful_sigterm():
            raise ValueError("training failed")
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.cluster
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX signal delivery")
def test_sigterm_after_real_ray_init_runs_cleanup(tmp_path):
    address = os.environ.get("ELASTIC_ROLLOUT_TEST_RAY_ADDRESS")
    if not address:
        pytest.skip("set ELASTIC_ROLLOUT_TEST_RAY_ADDRESS to an existing Ray cluster")
    pytest.importorskip("ray")
    marker = tmp_path / "closed"
    script = """
import sys, time
from pathlib import Path
import ray
from elastic_rollout.integrations.areal.lifecycle import graceful_sigterm
ray.init(address=sys.argv[1])
try:
    with graceful_sigterm():
        try:
            print('READY', flush=True)
            while True: time.sleep(1)
        finally:
            Path(sys.argv[2]).write_text('closed')
finally:
    ray.shutdown()
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script, address, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = b""
    try:
        deadline = time.monotonic() + 60
        while b"READY\n" not in output:
            assert proc.poll() is None, output.decode(errors="replace")
            assert time.monotonic() < deadline, output.decode(errors="replace")
            if select.select([proc.stdout], [], [], 1)[0]:
                output += os.read(proc.stdout.fileno(), 4096)
        proc.send_signal(signal.SIGTERM)
        remaining, _ = proc.communicate(timeout=30)
        assert proc.returncode == 143, (output + remaining).decode(errors="replace")
        assert marker.read_text() == "closed"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
