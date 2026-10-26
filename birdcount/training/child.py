"""Run a training tool as a child process that its parent can stop like Ctrl+C.

    python -m birdcount.training.child birdcount.training.train --epochs 10

The tool runs as `python -m <module>` would run it. When the stop file named
by the parent appears, or the parent process is gone, KeyboardInterrupt is
raised in the tool's main thread, so it stops the way it does on Ctrl+C:
training saves its state and writes summary.json, a video analysis writes out
what it sampled. Killing the process instead loses both.

Why a file: Ctrl+C cannot be sent to another process group on Windows,
Ctrl+Break would kill the DataLoader workers on the spot, and a thread
blocked reading a stdin pipe deadlocks the process as soon as it loads a
DLL (importing numpy is enough).

Only one interrupt is delivered, whichever comes first (the stop file or a
real Ctrl+C), so a second one cannot land in the middle of the tool's
clean-up. Parents start the child with `command()` and `env()`, and stop it
with `stop()`.
"""

from __future__ import annotations

import _thread
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path


STOP_FILE_ENV = "BIRDCOUNT_STOP_FILE"
PARENT_ENV = "BIRDCOUNT_PARENT_PID"
POLL_SECONDS = 0.25

_interrupted = threading.Event()


def command(module: str, *args: str) -> list[str]:
    """argv running `module` (a package with __main__, or a module) under this wrapper."""
    return [sys.executable, "-u", "-m", __name__, module, *args]


def env(stop_file: Path, base: dict | None = None) -> dict:
    """The child's environment: where its stop file will appear, and who its parent is."""
    stop_file.unlink(missing_ok=True)
    return {**(os.environ if base is None else base), STOP_FILE_ENV: str(stop_file), PARENT_ENV: str(os.getpid())}


def stop(stop_file: Path) -> None:
    """Ask the child to stop; returns at once."""
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()


def wait(proc: subprocess.Popen) -> int:
    """`proc.wait()` that a KeyboardInterrupt can break into (on Windows a plain
    wait blocks it until the child exits)."""
    while True:
        try:
            return proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass


def _parent_alive(pid: int) -> Callable[[], bool] | None:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return None
        return lambda: kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running

    def alive() -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True

    return alive


def _on_sigint(signum, frame) -> None:
    if _interrupted.is_set():
        return  # one interrupt is enough: let the tool finish cleaning up
    _interrupted.set()
    raise KeyboardInterrupt


def _watch(stop_file: str | None, parent_alive: Callable[[], bool] | None) -> None:
    while not _interrupted.is_set():
        if (stop_file and os.path.exists(stop_file)) or (parent_alive is not None and not parent_alive()):
            _thread.interrupt_main(signal.SIGINT)
            return
        time.sleep(POLL_SECONDS)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    stop_file = os.environ.get(STOP_FILE_ENV) or None
    parent = os.environ.get(PARENT_ENV, "")
    # The tool's own children (DataLoader workers, a sweep's runs) must not watch our files.
    os.environ.pop(STOP_FILE_ENV, None)
    os.environ.pop(PARENT_ENV, None)
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        if stop_file or parent:
            alive = _parent_alive(int(parent)) if parent.isdigit() else None
            threading.Thread(target=_watch, args=(stop_file, alive), name="stop-watch", daemon=True).start()
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
