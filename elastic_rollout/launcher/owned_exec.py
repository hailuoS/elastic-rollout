"""Keep a Linux child owned by its launcher across exec and abrupt parent exit."""

import ctypes
import os
import signal
import sys


def owned_command(argv):
    if sys.platform != "linux":
        return argv
    return [sys.executable, "-m", __name__, str(os.getpid()), *argv]


def main():
    owner = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0):  # PR_SET_PDEATHSIG survives exec.
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    # Parent death before prctl must not start an unowned service.
    if os.getppid() != owner:
        return 1
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
