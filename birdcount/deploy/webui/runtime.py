"""Start, stop and restart the runtime from the farm backend's page.

One operation at a time, on a thread of its own; the page polls /api/status
and sees it in `control`: {"busy": "start" | "stop" | "restart" | None,
"since", "step", "error"}. Whether the runtime is up is judged by the status
file's heartbeat and the pid it names, never by what this process started:
a runtime started by hand is stopped and restarted the same way.

Stop asks first (deploy.control's request file, the same orderly shutdown as
Terminate) and ends the process tree only when the runtime has not stopped
after STOP_TIMEOUT: one that no longer answers is what a restart is for.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from birdcount.deploy import control
from birdcount.deploy.webui import store


logger = logging.getLogger(__name__)

STOP_TIMEOUT = 60.0  # s for the orderly shutdown before the process tree is ended
KILL_TIMEOUT = 10.0
START_TIMEOUT = 420.0  # s to the first heartbeat: the model's warm-up (app.READY_TIMEOUT) and the cameras
POLL = 0.5


class RuntimeControl:
    def __init__(self, status_path: Path | None, *, start: Callable = control.start_runtime):
        self.status_path = status_path
        self._start = start
        self._lock = threading.Lock()
        self.busy: str | None = None
        self.since: float | None = None
        self.step = ""
        self.error: str | None = None
        self.proc = None  # the last runtime started from here, while it starts

    def state(self) -> dict:
        return {"busy": self.busy, "since": self.since, "step": self.step, "error": self.error}

    def running_pid(self) -> int | None:
        """The pid of a runtime that is still a process, from the status file; None when there is none."""
        st = store.read_status(self.status_path)
        pid = st.get("pid")
        return pid if st.get("state") in ("running", "gone", "unreadable") and control.alive(pid) else None

    def run(self, action: str) -> dict:
        """Begin `action` (start, stop, restart) and return at once; raises ValueError when it cannot begin."""
        if self.status_path is None:
            raise ValueError("STATUS_PATH is not set: the backend cannot see the runtime")
        if action not in ("start", "stop", "restart"):
            raise ValueError(f"unknown action {action!r}")
        with self._lock:
            if self.busy:
                raise ValueError(f"busy: {self.busy} is under way")
            pid = self.running_pid()
            if action == "start" and pid:
                raise ValueError("the runtime is already running")
            if action == "stop" and not pid:
                raise ValueError("the runtime is not running")
            self.busy, self.since, self.step, self.error = action, time.time(), "", None
        logger.info("Runtime %s requested from the page.", action)
        threading.Thread(target=self._work, args=(action, pid), name=f"runtime-{action}", daemon=True).start()
        return self.state()

    def _work(self, action: str, pid: int | None) -> None:
        try:
            if pid and action in ("stop", "restart"):
                self._stop(pid)
            if action in ("start", "restart"):
                self._start_and_wait()
        except Exception as e:  # shown on the page; the next action starts clean
            self.error = str(e)
            logger.warning("Runtime %s failed: %s", action, e)
        finally:
            self.busy, self.step = None, ""

    def _stop(self, pid: int) -> None:
        self.step = "stopping"
        control.request_stop(self.status_path, pid)
        if self._wait(lambda: not control.alive(pid), STOP_TIMEOUT):
            return
        logger.warning("Runtime %d did not stop in %.0f s; ending its processes.", pid, STOP_TIMEOUT)
        self.step = "ending its processes"
        control.force_stop(pid)
        if not self._wait(lambda: not control.alive(pid), KILL_TIMEOUT):
            raise RuntimeError(f"process {pid} is still there; end it in Task Manager")

    def _start_and_wait(self) -> None:
        self.step = "starting"
        begun = time.time()
        self.proc = self._start()

        def up() -> bool:
            st = store.read_status(self.status_path)
            return st.get("state") == "running" and (st.get("started") or 0) >= begun - 1

        def ended() -> bool:
            return self.proc.poll() is not None

        try:
            if self._wait(lambda: up() or ended(), START_TIMEOUT) and up():
                return
            if ended():
                code = self.proc.returncode
                why = {2: "configuration error, or another runtime is running"}.get(code, "it failed")
                raise RuntimeError(f"the runtime exited with code {code} ({why}); see logs/deploy.log")
            raise RuntimeError(f"no heartbeat {START_TIMEOUT:.0f} s after starting; see logs/deploy.log")
        finally:
            self.proc = None

    @staticmethod
    def _wait(done: Callable[[], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not done():
            if time.monotonic() > deadline:
                return False
            time.sleep(POLL)
        return True
