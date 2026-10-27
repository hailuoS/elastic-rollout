"""Actual exec and kernel parent-death delivery; no GPU or framework required."""

import os
import signal
import subprocess
import sys
import time

import pytest

from elastic_rollout.launcher.owned_exec import owned_command


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PR_SET_PDEATHSIG required")
@pytest.mark.parametrize("parent_exit", ["sigkill", "normal"])
def test_exec_child_receives_sigterm_when_launcher_disappears(tmp_path, parent_exit):
    ready, closed = tmp_path / "ready", tmp_path / "closed"
    child = (
        "import signal,time,os; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM,lambda *_: (Path({str(closed)!r}).touch(),exit(0))); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    parent = (
        "import subprocess,sys; from elastic_rollout.launcher.owned_exec import owned_command; "
        f"child=subprocess.Popen(owned_command([sys.executable,'-c',{child!r}])); "
        "print(child.pid,flush=True); sys.stdin.read()"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    child_pid = int(proc.stdout.readline())
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert proc.poll() is None
            assert time.monotonic() < deadline, "child did not exec"
            time.sleep(0.02)
        if parent_exit == "sigkill":
            proc.kill()
        else:
            proc.stdin.close()
        proc.wait(timeout=10)
        deadline = time.monotonic() + 10
        while not closed.exists():
            assert time.monotonic() < deadline, "orphan did not receive SIGTERM"
            time.sleep(0.02)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.stdout.close()
        if not proc.stdin.closed:
            proc.stdin.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PR_SET_PDEATHSIG required")
def test_owner_exit_before_registration_does_not_exec(tmp_path):
    marker = tmp_path / "unexpected"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "elastic_rollout.launcher.owned_exec",
            "-1",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        timeout=10,
    )
    assert result.returncode == 1 and not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PR_SET_PDEATHSIG required")
def test_exec_preserves_the_child_exit_code():
    result = subprocess.run(
        owned_command([sys.executable, "-c", "raise SystemExit(7)"]), timeout=10
    )
    assert result.returncode == 7


def test_other_platforms_keep_the_original_command(monkeypatch):
    monkeypatch.setattr("elastic_rollout.launcher.owned_exec.sys.platform", "win32")
    argv = ["python", "-m", "module"]
    assert owned_command(argv) is argv
