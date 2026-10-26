"""Long-running background servers started from the UI (currently Label Studio).

Services deliberately live outside `RunManager`: that one enforces a single
active run so two trainings cannot fight over the GPU, and an annotation server
that stays up for hours must never consume that slot. They reuse `Run` for the
process plumbing — merged output, tee'd log, process-tree kill — but are tracked
here and are not part of the run history.

The launch command mirrors tools/starter.sh's `label-studio` branch, so starting
from the UI and from the shell produce the same server.
"""

import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Optional

from .runs import ROOT, Run


_IS_WINDOWS = os.name == "nt"

LABEL_STUDIO = "label_studio"
NGROK = "ngrok"

# ngrok's local inspector, handy for confirming the tunnel really bound. Spelled
# 127.0.0.1 rather than localhost: the probe usually finds nothing there, and
# "localhost" makes it wait out an IPv6 attempt before retrying over IPv4.
NGROK_INSPECTOR_PORT = "4040"
NGROK_INSPECTOR = f"http://127.0.0.1:{NGROK_INSPECTOR_PORT}"

# Command-line fragments that identify a process as one of ours to adopt. A
# server started from a shell is still stoppable from the UI, but only after its
# command line proves it really is that server — 8080 could just as easily be
# someone else's dev server, and killing that would be a bad afternoon.
ADOPT_PATTERNS = {
    LABEL_STUDIO: ("label-studio", "label_studio"),
    NGROK: ("ngrok",),
}

# Defaults match tools/starter.sh (LS_PORT / LS_DATA_DIR / LS_LOCAL_FILES_ROOT).
DEFAULT_PORT = os.getenv("LS_PORT", "8080")
DEFAULT_DATA_DIR = os.getenv("LS_DATA_DIR") or str((ROOT.parent / "data").resolve())
DEFAULT_LOCAL_FILES_ROOT = os.getenv("LS_LOCAL_FILES_ROOT") or str((ROOT.parent / "data").resolve())


def strip_scheme(domain: str) -> str:
    return domain.removeprefix("https://").removeprefix("http://").rstrip("/")


def _resolve(program: str) -> str:
    path = shutil.which(program)
    if not path:
        raise RuntimeError(f"`{program}` is not on PATH — install it or start it from a shell.")
    return path


def ngrok_command(domain: str, port: str) -> tuple[list[str], dict]:
    """argv + env for the tunnel starter.sh tells you to run in another terminal."""
    domain = strip_scheme(domain)
    if not domain:
        raise RuntimeError("No public domain configured — set LABEL_STUDIO_PUBLIC_URL in .env.")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # ngrok's fullscreen TUI would emit escape sequences into the log; --log
    # makes it write plain lines to stdout instead.
    argv = [_resolve("ngrok"), "http", f"--url={domain}", str(port), "--log=stdout", "--log-format=term"]
    return argv, env


def label_studio_command(public_domain: str = "") -> tuple[list[str], dict]:
    """argv + env for `label-studio start`, matching tools/starter.sh.

    With `public_domain` set, Label Studio is told to trust that origin so the
    ngrok tunnel can serve it — the same two variables starter.sh exports.
    """
    executable = _resolve("label-studio")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = DEFAULT_LOCAL_FILES_ROOT

    domain = strip_scheme(public_domain)
    if domain:
        env["CSRF_TRUSTED_ORIGINS"] = f"https://{domain}"
        env["LABEL_STUDIO_HOST"] = f"https://{domain}"

    argv = [executable, "start", "--data-dir", DEFAULT_DATA_DIR, "-p", DEFAULT_PORT]
    return argv, env


def _listening_pids(port: str) -> list[int]:
    """PIDs with a LISTEN socket on `port`."""
    pids: set[int] = set()
    try:
        if _IS_WINDOWS:
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
                    if parts[1].rsplit(":", 1)[-1] == str(port):
                        pids.add(int(parts[4]))
        else:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
            pids.update(int(value) for value in out.split())
    except (OSError, ValueError):
        return []
    return sorted(pids)


def _cmdline(pid: int) -> str:
    try:
        if _IS_WINDOWS:
            script = f'(Get-CimInstance Win32_Process -Filter "ProcessId={pid}").CommandLine'
            return subprocess.run(
                ["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True
            ).stdout.strip()
        return subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def external_processes(name: str, port: str) -> list[dict]:
    """Processes listening on `port` whose command line identifies them as `name`.

    Anything on the port that does not match is reported with `matches=False` so
    the caller can explain why it refused to kill it.
    """
    patterns = ADOPT_PATTERNS.get(name, ())
    found = []
    for pid in _listening_pids(port):
        command = _cmdline(pid)
        found.append(
            {
                "pid": pid,
                "command": command,
                "matches": any(p in command.lower() for p in patterns),
            }
        )
    return found


def kill_tree(pid: int) -> bool:
    """Terminate a process and its children. True when the kill command succeeded."""
    try:
        if _IS_WINDOWS:
            done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            return done.returncode == 0
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.25)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
        os.kill(pid, signal.SIGKILL)
        return True
    except (OSError, ProcessLookupError):
        return False


class ServiceManager:
    """One optional background process per service name."""

    def __init__(self):
        self._services: dict[str, Run] = {}
        self._lock = threading.Lock()

    def start(self, name: str, argv: list[str], env: dict, values: Optional[dict] = None) -> Run:
        with self._lock:
            current = self._services.get(name)
            if current is not None and current.running:
                raise RuntimeError(f"{name} is already running (pid in the log); stop it first")
            # "service-" prefixed ids keep these out of the restored run history.
            run_id = f"service-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
            service = Run(name, argv, values or {}, run_id=run_id, env=env)
            self._services[name] = service
        service.start()
        return service

    def get(self, name: str) -> Optional[Run]:
        return self._services.get(name)

    def stop(self, name: str) -> Optional[Run]:
        service = self._services.get(name)
        if service is not None:
            service.stop()
        return service

    def status(self, name: str) -> dict:
        service = self._services.get(name)
        if service is None:
            return {"name": name, "state": "stopped", "started": False}
        data = service.summary()
        data.update({"name": name, "state": service.status, "started": True})
        return data

    def stop_all(self) -> None:
        for name in list(self._services):
            self.stop(name)
