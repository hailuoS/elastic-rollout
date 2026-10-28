"""Signal scope for a standalone AReaL training driver."""

import signal
from contextlib import contextmanager


@contextmanager
def graceful_sigterm():
    """Install after ray.init so SIGTERM unwinds the trainer's context manager.

    Keep this scope around construction, training and close. The caller owns
    ray.shutdown and the separately launched Core's lifecycle.
    """
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, _frame):
        # A second termination request must not interrupt resource cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
