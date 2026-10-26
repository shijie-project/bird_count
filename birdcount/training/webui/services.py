"""Label Studio and its ngrok tunnel: started, watched and stopped from the panel.

Services are not jobs: Label Studio stays up for hours of annotating and must
not hold the GPU slot. Each is a plain process whose output goes to a log
under the panel's folder.

A server started from a shell can be stopped here too, but only after its
command line proves it is that server: anything else listening on the port
is left alone.

Probes use urllib; the HTTP client loads only to list projects.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from birdcount.training.webui.jobs import kill_tree, spawn
from birdcount.training.webui.settings import WebuiSettings


LABEL_STUDIO = "label_studio"
NGROK = "ngrok"
SERVICES = (LABEL_STUDIO, NGROK)

# ngrok's local inspector answers while a tunnel runs, whoever started it.
NGROK_INSPECTOR_PORT = 4040
NGROK_INSPECTOR = f"http://127.0.0.1:{NGROK_INSPECTOR_PORT}"
# A local server answers in milliseconds, but Windows takes ~2 s to give up on a
# closed port (it retries the connect), and the page waits on this probe.
PROBE_TIMEOUT = {LABEL_STUDIO: 1.0, "public": 2.5, NGROK: 0.3}
PROJECTS_TTL = 60.0

# Command-line fragments that identify a process listening on the port as ours to stop.
ADOPT_PATTERNS = {LABEL_STUDIO: ("label-studio", "label_studio"), NGROK: ("ngrok",)}

_IS_WINDOWS = os.name == "nt"


def _executable(program: str) -> str:
    path = shutil.which(program)
    if not path:
        raise RuntimeError(f"`{program}` is not on PATH; install it, or start it from a shell")
    return path


def label_studio_command(s: WebuiSettings, public: bool) -> tuple[list[str], dict]:
    env = {
        **os.environ,
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(s.files_root),
    }
    if public:
        if not s.public_domain:
            raise RuntimeError("no public domain: set LABEL_STUDIO_PUBLIC_URL in .env")
        env["CSRF_TRUSTED_ORIGINS"] = f"https://{s.public_domain}"
        env["LABEL_STUDIO_HOST"] = f"https://{s.public_domain}"
    argv = [
        _executable("label-studio"),
        "start",
        "--data-dir",
        str(s.ls_data_dir),
        "-p",
        str(s.ls_port),
        "--no-browser",
    ]
    return argv, env


def ngrok_command(s: WebuiSettings) -> tuple[list[str], dict]:
    if not s.public_domain:
        raise RuntimeError("no public domain: set LABEL_STUDIO_PUBLIC_URL in .env")
    # --log=stdout: plain lines instead of ngrok's full-screen display
    argv = [
        _executable("ngrok"),
        "http",
        f"--url={s.public_domain}",
        str(s.ls_port),
        "--log=stdout",
        "--log-format=term",
    ]
    return argv, dict(os.environ)


def probe(url: str, timeout: float) -> int | None:
    """HTTP status of a GET, or None when nothing answers."""
    import urllib.error
    import urllib.request

    # 127.0.0.1, not localhost: when nothing listens, "localhost" first waits out IPv6
    url = url.replace("://localhost", "://127.0.0.1", 1)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except (OSError, ValueError):
        return None


def listening_pids(port: int) -> list[int]:
    pids: set[int] = set()
    try:
        if _IS_WINDOWS:
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split()
                listening = len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING"
                if listening and parts[1].rsplit(":", 1)[-1] == str(port):
                    pids.add(int(parts[4]))
        else:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
            pids.update(int(v) for v in out.split())
    except (OSError, ValueError):
        return []
    return sorted(pids)


def command_line(pid: int) -> str:
    try:
        if _IS_WINDOWS:
            script = f'(Get-CimInstance Win32_Process -Filter "ProcessId={pid}").CommandLine'
            argv = ["powershell", "-NoProfile", "-Command", script]
        else:
            argv = ["ps", "-p", str(pid), "-o", "args="]
        return subprocess.run(argv, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


class Service:
    def __init__(self, name: str, log_path: Path):
        self.name = name
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.started: float | None = None
        self.public = False

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class Services:
    def __init__(self, settings: WebuiSettings, folder: Path):
        self.s = settings
        self.folder = folder
        self._services = {name: Service(name, folder / f"{name}.log") for name in SERVICES}
        self._lock = threading.Lock()
        self._projects: tuple[float, dict] | None = None
        self._client = None

    def port(self, name: str) -> int:
        return NGROK_INSPECTOR_PORT if name == NGROK else self.s.ls_port

    def start(self, name: str, public: bool = False) -> dict:
        argv, env = label_studio_command(self.s, public) if name == LABEL_STUDIO else ngrok_command(self.s)
        with self._lock:
            service = self._services[name]
            if service.running:
                raise RuntimeError(f"{name} is already running")
            self.folder.mkdir(parents=True, exist_ok=True)
            service.log_path.write_text(f"$ {subprocess.list2cmdline(argv)}\n", encoding="utf-8")
            service.proc = spawn(argv, service.log_path, env)
            service.started, service.public = time.time(), public
        return self.state(name)

    def stop(self, name: str) -> dict:
        """Stop ours, else the matching process listening on the service's port."""
        service = self._services[name]
        killed, refused = [], []
        if service.running:
            kill_tree(service.proc.pid)
            service.proc.wait(timeout=10)
            killed.append({"pid": service.proc.pid, "owned": True})
        else:
            for pid in listening_pids(self.port(name)):
                command = command_line(pid)
                entry = {"pid": pid, "command": command}
                if any(p in command.lower() for p in ADOPT_PATTERNS[name]):
                    if kill_tree(pid):
                        killed.append(entry)
                else:
                    refused.append(entry)
        return {**self.state(name), "killed": killed, "refused": refused}

    def stop_all(self) -> None:
        for service in self._services.values():
            if service.running:
                kill_tree(service.proc.pid)

    def state(self, name: str) -> dict:
        service = self._services[name]
        return {
            "name": name,
            "owned": service.running,
            "public": service.public,
            "started": service.started if service.running else None,
            "exit": None if service.proc is None or service.running else service.proc.returncode,
        }

    def status(self, public: bool) -> dict:
        """Both services, probed in parallel: whether Label Studio and the tunnel answer."""
        local = self.s.label_studio_url.rstrip("/")
        public_url = f"https://{self.s.public_domain}" if self.s.public_domain else ""
        url = public_url if public else local
        codes: dict[str, int | None] = {}

        def run(name: str, target: str) -> None:
            codes[name] = probe(target, PROBE_TIMEOUT["public" if public and name == LABEL_STUDIO else name])

        probes = [threading.Thread(target=run, args=(NGROK, NGROK_INSPECTOR + "/api/tunnels"))]
        if url:
            probes.append(threading.Thread(target=run, args=(LABEL_STUDIO, url)))
        for t in probes:
            t.start()
        for t in probes:
            t.join()
        code, tunnel_code = codes.get(LABEL_STUDIO), codes.get(NGROK)
        return {
            "local_url": local,
            "public_url": public_url,
            "url": url,
            # < 400 only: an idle ngrok domain answers 404 from ngrok itself
            "reachable": code is not None and code < 400,
            "detail": "not configured" if not url else ("no answer" if code is None else f"HTTP {code}"),
            "tunnel": tunnel_code is not None and tunnel_code < 500,
            "inspector": NGROK_INSPECTOR,
            "data_dir": str(self.s.ls_data_dir),
            "files_root": str(self.s.files_root),
            "services": {name: self.state(name) for name in SERVICES},
        }

    def projects(self, refresh: bool = False) -> dict:
        """The server's projects, for the project picker (kept PROJECTS_TTL s)."""
        if not refresh and self._projects and time.monotonic() - self._projects[0] < PROJECTS_TTL:
            return self._projects[1]
        token = self.s.label_studio_api_key.get_secret_value()
        if not token:
            return {"items": [], "error": "LABEL_STUDIO_API_KEY is not set in .env"}
        import httpx

        from birdcount.training.labeling.client import LabelStudio, LabelStudioError

        try:
            if self._client is None:
                self._client = LabelStudio(self.s.label_studio_url, token, timeout=5.0)
            data = self._client.request("GET", "/api/projects/", params={"page_size": 1000})
        except (LabelStudioError, httpx.HTTPError) as e:
            self._client = None
            return {"items": [], "error": str(e)}
        projects = data.get("results", []) if isinstance(data, dict) else data
        out = {
            "items": [
                {
                    "value": str(p["id"]),
                    "label": f"{p['id']} · {p['title']}",
                    "detail": f"{p.get('task_number', '?')} tasks",
                }
                for p in projects
            ]
        }
        self._projects = (time.monotonic(), out)
        return out
