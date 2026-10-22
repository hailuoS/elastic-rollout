import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from elastic_rollout.launcher import supervisor


@pytest.fixture(autouse=True)
def posix_signals(monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "signal",
        SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, SIGKILL=9, signal=Mock()),
    )


@pytest.mark.parametrize("stubborn", [False, True])
def test_cleanup_escalates_and_waits_for_group_exit(monkeypatch, stubborn):
    process = Mock(pid=123)
    kill = Mock()
    monkeypatch.setattr(supervisor, "os", SimpleNamespace(killpg=kill))
    wait = Mock(side_effect=[False, True] if stubborn else [True])
    monkeypatch.setattr(supervisor, "_wait_group_exit", wait)
    supervisor._cleanup(process)
    assert kill.call_args_list[0].args == (123, signal.SIGTERM)
    if stubborn:
        assert kill.call_args_list[1].args == (123, 9)
    process.wait.assert_called_once_with(timeout=1)


def test_surviving_group_never_acknowledges_cleanup(monkeypatch):
    monkeypatch.setattr(supervisor, "os", SimpleNamespace(killpg=Mock()))
    monkeypatch.setattr(supervisor, "_wait_group_exit", lambda *_: False)
    with pytest.raises(RuntimeError, match="survived cleanup"):
        supervisor._cleanup(Mock(pid=123))


@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_signal_during_spawn_requires_verified_cleanup(monkeypatch, cleanup_failed):
    handlers = {}
    monkeypatch.setattr(
        supervisor.signal, "signal", lambda sig, handler: handlers.update({sig: handler})
    )
    write = Mock()
    monkeypatch.setattr(
        supervisor,
        "os",
        SimpleNamespace(
            name="posix",
            environ={"ELASTIC_ROLLOUT_CLEANUP_FD": "9"},
            write=write,
            getpid=lambda: 100,
        ),
    )
    monkeypatch.setattr(supervisor.threading, "Thread", Mock())
    process = Mock(returncode=0)
    process.poll.return_value = None

    def spawn(*args, **kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", spawn)
    cleanup = Mock(side_effect=RuntimeError("group survived") if cleanup_failed else None)
    monkeypatch.setattr(supervisor, "_cleanup", cleanup)
    if cleanup_failed:
        with pytest.raises(RuntimeError, match="group survived"):
            supervisor.main()
        write.assert_not_called()
    else:
        assert supervisor.main() == 0
        write.assert_called_once_with(9, b"CLEAN\n")
    cleanup.assert_called_once_with(process)


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux process groups")
@pytest.mark.parametrize("action", ["eof", "sigterm", "stubborn"])
def test_real_guardian_cleans_child_group(tmp_path, action):
    marker = tmp_path / "child.pid"
    child_script = (
        "import os, time, signal; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if action == "stubborn" else "")
        + f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    reader, writer = os.pipe()
    env = os.environ | {"ELASTIC_ROLLOUT_CLEANUP_FD": str(writer)}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "elastic_rollout.launcher.supervisor",
            sys.executable,
            "-c",
            child_script,
        ],
        stdin=subprocess.PIPE,
        env=env,
        pass_fds=(writer,),
        start_new_session=True,
    )
    os.close(writer)
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert proc.poll() is None
            if time.monotonic() >= deadline:
                pytest.fail("child did not start")
            time.sleep(0.05)
        child_pid = int(marker.read_text())
        assert os.getpgid(child_pid) == child_pid
        if action in ("eof", "stubborn"):
            proc.stdin.close()
        else:
            proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 0
        assert os.read(reader, 64) == b"CLEAN\n"
        with pytest.raises(ProcessLookupError):
            os.killpg(child_pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.stdin.close()
        os.close(reader)
