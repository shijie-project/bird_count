"""The panel's HTTP server: the page, its JSON API, and the run files it shows.

Python's own threading HTTP server is enough for one person's local panel
and keeps the process small (v2's FastAPI + Label Studio SDK server idled at
~200 MB). The server never imports torch, numpy or cv2: tools run as jobs.

Requests are checked before they are routed:
* the Host header must name this machine, so a web page cannot reach the
  panel through DNS rebinding;
* anything but GET must come from the panel's own page (Origin), so another
  site open in the browser cannot start, stop or delete anything;
* files are served only from the data, output, checkpoint and mask folders
  (and folders the panel's own jobs write), and only of the kinds the page
  shows — never `.env` or source code.
"""

from __future__ import annotations

import email.utils
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from birdcount.core.paths import IMAGE_SUFFIXES, PROJECT_ROOT, VIDEO_SUFFIXES, resolve_path
from birdcount.training import child, runs
from birdcount.training.webui import results
from birdcount.training.webui.dataset import Dataset
from birdcount.training.webui.jobs import Busy, Jobs
from birdcount.training.webui.services import SERVICES, Services
from birdcount.training.webui.settings import WebuiSettings
from birdcount.training.webui.tools import PASSTHROUGH, TOOLS, Specs, Tool, build_args


logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_JSON_BODY = 2 * 2**20
MAX_UPLOAD = {"json": 200 * 2**20, "video": 8 * 2**30}
UPLOAD_SUFFIXES = {"json": (".json",), "video": VIDEO_SUFFIXES}
SERVED_SUFFIXES = {*IMAGE_SUFFIXES, ".webp", *VIDEO_SUFFIXES, ".webm", ".csv", ".json", ".jsonl", ".log", ".txt"}
CONTENT_TYPES = {  # the Windows registry can map these to text/plain, which breaks modules
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".jsonl": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".mkv": "video/x-matroska",
}
PAGE_RUNS = {"train": ("train", "sweep"), "eval": ("eval",), "video": ("video",)}
PICKER_LIMIT = 500


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _rel(path: Path) -> str:
    """How a path is spelled in a command line: relative to the project when it can be."""
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:  # another drive
        return str(path)


def _age_sorted(paths, limit: int = PICKER_LIMIT) -> list[Path]:
    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(paths, key=mtime, reverse=True)[:limit]


class Panel:
    """Everything the request handlers share."""

    def __init__(self, settings: WebuiSettings, *, restartable: bool = True):
        self.s = settings
        self.folder = settings.output_root / "webui"
        self.jobs = Jobs(self.folder / "jobs")
        self.services = Services(settings, self.folder / "services")
        self.specs = Specs(self.folder / "tools-cache.json")
        self.dataset = Dataset(settings.data_root)
        self.instance = uuid.uuid4().hex
        self.restartable = restartable
        self.restart_requested = threading.Event()
        self.server: ThreadingHTTPServer | None = None

    # --- file access ---------------------------------------------------------------------------

    def roots(self) -> list[Path]:
        s = self.s
        found = [s.data_root, s.output_root, s.files_root]
        if s.model_path is not None:
            found.append(s.model_path.parent)
        if s.mask_dir is not None:
            found.append(s.mask_dir)
        found += [j.run_dir for j in self.jobs.all() if j.run_dir is not None]
        return [p.resolve() for p in found]

    def allowed(self, path: Path) -> bool:
        path = path.resolve()
        if path.name.lower() == ".env" or path.is_relative_to(PROJECT_ROOT / "birdcount"):
            return False
        return any(path.is_relative_to(root) for root in self.roots())

    def run_path(self, ref: str) -> tuple[Path, str]:
        path = Path(ref) if Path(ref).is_absolute() else self.s.output_root / ref
        path = path.resolve()
        kind = runs.kind_of(path)
        if kind is None or not self.allowed(path):
            raise HttpError(404, f"no run at {ref}")
        return path, kind

    def ref(self, path: Path) -> str:
        """How the page names a run: its path under OUTPUT_ROOT, else the absolute path.
        Paths here are already absolute and resolved (settings, `runs.find`, job records)."""
        try:
            return path.relative_to(self.s.output_root).as_posix()
        except ValueError:
            return str(path)

    # --- tools and jobs ------------------------------------------------------------------------

    def plan(self, tool: Tool, values: dict) -> tuple[dict, Path | None]:
        """Pick the run folder of a run-writing tool, so the panel knows where its results land.

        A folder typed into the form wins. Folder names follow `birdcount.training.runs`,
        the same rules the tools apply when run from a shell.
        """
        values = {k: v for k, v in values.items() if v not in (None, "", [])}
        root = self.s.output_root
        given = values.get("output_dir")
        if tool.run == "train":
            if values.get("resume"):
                return values, resolve_path(values["resume"])
            if not given:
                values["output_dir"] = _rel(runs.train_dir(root, str(values.get("name", ""))))
        elif tool.run == "sweep":
            given = values.get("sweep_dir")
            if not given:
                values["sweep_dir"] = _rel(runs.sweep_dir(root))
            return values, resolve_path(values["sweep_dir"])
        elif tool.run == "eval" and not given:
            checkpoint = values.get("ckpt") or self.s.model_path or "model"
            split = values.get("split") or self.specs_default(tool, "split")
            values["output_dir"] = _rel(runs.eval_dir(root, checkpoint, str(split)))
        elif tool.run == "video" and not given:
            videos = [v for v in str(values.get("videos", "")).splitlines() if v.strip()]
            stamp = runs.stamp()
            if len(videos) == 1 and resolve_path(videos[0].strip()).is_file():
                values["output_dir"] = _rel(runs.video_dir(root, Path(videos[0].strip()), stamp))
            else:  # several videos: one folder each inside this one
                values["output_dir"] = _rel(root / "video" / f"{stamp}-batch")
        if tool.run is None:
            return values, None
        return values, resolve_path(values["output_dir"])

    def specs_default(self, tool: Tool, dest: str):
        for group in self.specs.get(tool.key)["groups"]:
            for opt in group["options"]:
                if opt["dest"] == dest:
                    return opt["default"]
        return None

    def command(self, body: dict) -> tuple[Tool, list[str], Path | None, dict]:
        tool = TOOLS.get(body.get("tool", ""))
        if tool is None:
            raise HttpError(404, f"unknown tool {body.get('tool')!r}")
        values = body.get("values") or {}
        if not isinstance(values, dict):
            raise HttpError(400, "values must be an object")
        try:
            values, run_dir = self.plan(tool, values)
            args = build_args(tool, self.specs.get(tool.key), values)
        except ValueError as e:
            raise HttpError(400, str(e)) from None
        return tool, args, run_dir, values

    def runs_of(self, page: str) -> list[dict]:
        """Run folders and jobs of one page, newest first."""
        items, claimed = [], set()
        by_dir = self.jobs.by_run_dir()

        def add(path: Path, kind: str) -> None:
            job = by_dir.get(path) or by_dir.get(path.parent)
            if job is not None:
                claimed.add(job.id)
            info = results.overview_cached(path, kind, job is not None and job.running)
            items.append({"type": "run", "ref": self.ref(path), "job": job.to_json() if job else None, **info})

        seen = set()
        for kind in PAGE_RUNS.get(page, ()):
            for path in runs.find(self.s.output_root, kind):
                seen.add(path)
                add(path, kind)
        for job in self.jobs.all():
            if job.record["page"] != page or job.id in claimed:
                continue
            if job.run_dir is not None and job.run_dir in seen:
                continue  # an earlier job of a listed run (stopped, then resumed)
            inside = [p for p in runs.under(job.run_dir) if p not in seen] if job.run_dir else []
            for path in inside:  # a run folder outside the default layout
                add(path, runs.kind_of(path))
            if not inside:
                items.append({"type": "job", "name": job.record["tool"], **job.to_json()})
        return sorted(items, key=lambda i: i["started"], reverse=True)

    # --- pickers ---------------------------------------------------------------------------------

    def picker(self, kind: str, query: dict) -> dict:
        s = self.s
        if kind == "ls_project":
            return self.services.projects(refresh=query.get("refresh") == "1")
        if kind == "split":
            return {
                "items": [
                    {"value": d["name"], "label": d["name"], "detail": f"{d['images']} images"}
                    for d in self.dataset.splits()
                ]
            }
        paths: list[Path] = []
        if kind == "checkpoint":
            folders = [s.output_root / "train"] + ([s.model_path.parent] if s.model_path else [])
            paths = [p for f in folders if f.is_dir() for p in f.rglob("*.pth")]
            items = [self._file_item(p) for p in _age_sorted(set(paths))]
            if s.model_path is not None:
                items.insert(0, {**self._file_item(s.model_path), "detail": "MODEL_PATH (the default)"})
            return {"items": items}
        if kind == "video":
            paths = [p for p in s.data_root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES]
            paths += list((self.folder / "uploads").glob("*")) if (self.folder / "uploads").is_dir() else []
        elif kind == "json":
            folders = [s.data_root / "annotations", s.data_root / "export", self.folder / "uploads"]
            paths = [p for f in folders if f.is_dir() for p in f.glob("*.json")]
        elif kind == "blobs":
            paths = list((s.output_root / "eval").glob("*/*/blobs.json"))
        elif kind == "mask":
            paths = list(s.mask_dir.glob("*.png")) if s.mask_dir and s.mask_dir.is_dir() else []
        elif kind == "folder":
            paths = [p for p in (s.data_root / "images").glob("*") if p.is_dir()]
        elif kind in ("train_run", "sweep_run"):
            paths = runs.find(s.output_root, "train" if kind == "train_run" else "sweep")
        else:
            raise HttpError(404, f"unknown picker {kind!r}")
        return {"items": [self._file_item(p) for p in _age_sorted(paths)]}

    def _file_item(self, path: Path) -> dict:
        try:
            st = path.stat()
            detail = f"{st.st_size / 1e6:.1f} MB · {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"
            if path.is_dir():
                detail = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
        except OSError:
            detail = "missing"
        for root in (self.s.output_root, self.s.data_root):
            if path.is_relative_to(root):
                label = path.relative_to(root).as_posix()
                break
        else:
            label = path.name
        return {"value": _rel(path), "label": label, "detail": detail}

    # --- lifecycle ---------------------------------------------------------------------------------

    def schedule_stop(self, restart: bool) -> dict:
        if restart and not self.restartable:
            raise HttpError(409, "this panel cannot restart itself")
        if restart:
            self.restart_requested.set()

        def stop() -> None:
            time.sleep(0.3)  # let the reply reach the page first
            if self.server is not None:
                self.server.shutdown()

        threading.Thread(target=stop, daemon=True).start()
        return {"instance": self.instance, "jobs": [j.id for j in self.jobs.running()]}

    def close(self) -> None:
        """Stop what the panel started: running jobs (gracefully), then the services."""
        self.jobs.stop_all()
        self.services.stop_all()


# --- request handling -------------------------------------------------------------------------------


def _hostname(host: str) -> str:
    if host.startswith("["):
        return host[1 : host.find("]")].lower()
    return host.rsplit(":", 1)[0].lower()


def _local_host(name: str, bind: str) -> bool:
    if name in ("localhost", bind.lower(), socket.gethostname().lower()):
        return True
    try:
        ipaddress.ip_address(name)  # an address, not a name a page's DNS could re-point
        return True
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "birdcount-panel"
    panel: Panel  # set on the subclass made by make_server

    # --- plumbing ------------------------------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # the default logs every poll
        pass

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            self._guard(method)
            route = ROUTES.get((method, url.path))
            if route is None:
                for (m, pattern), fn in PATTERN_ROUTES:
                    if m == method and (match := pattern.fullmatch(url.path)):
                        route = lambda h, q, fn=fn, match=match: fn(h, q, *match.groups())  # noqa: E731
                        break
            if route is None:
                raise HttpError(404, f"no route {method} {url.path}")
            result = route(self, query)
            if result is not None:
                self._send_json(result)
        except HttpError as e:
            self._send_json({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            logger.exception("%s %s failed", method, self.path)
            self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _guard(self, method: str) -> None:
        host = self.headers.get("Host", "")
        if not _local_host(_hostname(host), self.server.server_address[0]):
            raise HttpError(403, "unexpected Host header")
        if method != "GET":
            origin = self.headers.get("Origin")
            if (origin and origin != f"http://{host}") or self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise HttpError(403, "requests must come from the panel's own page")

    def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, separators=(",", ":"), default=str).encode()
        self._send(status, body, "application/json", {"Cache-Control": "no-store"})

    def json_body(self) -> dict:
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise HttpError(415, "expected application/json")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_JSON_BODY:
            raise HttpError(413, "request too large")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise HttpError(400, "invalid JSON") from None
        if not isinstance(data, dict):
            raise HttpError(400, "expected a JSON object")
        return data

    def send_file(self, path: Path, *, cache: str = "no-cache") -> None:
        """A file, with Last-Modified revalidation and byte ranges (video seeking)."""
        st = path.stat()
        last_modified = email.utils.formatdate(st.st_mtime, usegmt=True)
        if self.headers.get("If-Modified-Since") == last_modified:
            self.send_response(304)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        suffix = path.suffix.lower()
        content_type = CONTENT_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size, start, end = st.st_size, 0, st.st_size - 1
        status = 200
        if (rng := re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))) and size:
            if rng[1]:
                start, end = int(rng[1]), min(int(rng[2]) if rng[2] else size - 1, size - 1)
            elif rng[2]:  # the last N bytes
                start = max(0, size - int(rng[2]))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


# --- routes -------------------------------------------------------------------------------------------


def page(h: Handler, q: dict) -> None:
    h.send_file(STATIC_DIR / "index.html")


def static(h: Handler, q: dict, name: str) -> None:
    path = (STATIC_DIR / name).resolve()
    if not path.is_relative_to(STATIC_DIR) or not path.is_file():
        raise HttpError(404, "no such file")
    h.send_file(path)


def health(h: Handler, q: dict) -> dict:
    p = h.panel
    return {"ok": True, "instance": p.instance, "restartable": p.restartable}


def shutdown(h: Handler, q: dict) -> dict:
    h.json_body()
    return h.panel.schedule_stop(restart=False)


def restart(h: Handler, q: dict) -> dict:
    h.json_body()
    return h.panel.schedule_stop(restart=True)


def tools(h: Handler, q: dict) -> dict:
    return {
        "tools": [
            {"key": t.key, "label": t.label, "page": t.page, "blurb": t.blurb, "gpu": t.gpu, "run": t.run}
            for t in TOOLS.values()
        ]
    }


def tool_spec(h: Handler, q: dict, key: str) -> dict:
    tool = TOOLS.get(key)
    if tool is None:
        raise HttpError(404, f"unknown tool {key!r}")
    try:
        spec = h.panel.specs.get(key)
    except RuntimeError as e:
        raise HttpError(500, str(e)) from None
    return {
        **spec,
        "key": key,
        "label": tool.label,
        "blurb": tool.blurb,
        "module": tool.module,
        "command": tool.command,
        "pickers": tool.pickers,
        "passthrough": tool.passthrough,
        "passthrough_key": PASSTHROUGH,
    }


def preview(h: Handler, q: dict) -> dict:
    try:
        tool, args, run_dir, _ = h.panel.command(h.json_body())
    except HttpError as e:
        if e.status != 400:
            raise
        return {"error": str(e)}  # an unfinished form is not a failed request
    return {
        "command": " ".join(["python", "-m", tool.module, *(f'"{a}"' if " " in a else a for a in args)]),
        "run_dir": str(run_dir) if run_dir else None,
    }


def start_job(h: Handler, q: dict) -> dict:
    p = h.panel
    tool, args, run_dir, values = p.command(h.json_body())
    try:
        job = p.jobs.start(tool.key, tool.page, child.command(tool.module, *args), run_dir=run_dir, gpu=tool.gpu)
    except Busy as e:
        raise HttpError(409, str(e)) from None
    # The run to show: the folder itself, unless it will hold one run per video.
    videos = [v for v in str(values.get("videos", "")).splitlines() if v.strip()]
    single = tool.run != "video" or (len(videos) == 1 and resolve_path(videos[0].strip()).is_file())
    return job.to_json() | {"ref": p.ref(run_dir) if run_dir and single else None}


def list_jobs(h: Handler, q: dict) -> dict:
    jobs = h.panel.jobs
    return {"jobs": [j.to_json() for j in jobs.all()], "running": [j.id for j in jobs.running()]}


def stop_job(h: Handler, q: dict, job_id: str) -> dict:
    h.json_body()
    job = h.panel.jobs.get(job_id)
    if job is None:
        raise HttpError(404, f"no job {job_id}")
    job.stop()
    return job.to_json()


def clear_jobs(h: Handler, q: dict) -> dict:
    h.json_body()
    return {"removed": h.panel.jobs.clear()}


def list_runs(h: Handler, q: dict) -> dict:
    return {"items": h.panel.runs_of(q.get("page", "")), "running": [j.id for j in h.panel.jobs.running()]}


def run_detail(h: Handler, q: dict) -> dict:
    p = h.panel
    path, kind = p.run_path(q.get("ref", ""))
    by_dir = p.jobs.by_run_dir()
    job = by_dir.get(path) or by_dir.get(path.parent)
    try:
        offset = int(q.get("offset", 0))
    except ValueError:
        raise HttpError(400, "bad offset") from None
    data = results.detail(path, kind, job is not None and job.running, offset)
    return data | {"ref": p.ref(path), "job": job.to_json() if job else None}


def log(h: Handler, q: dict) -> dict:
    """A job's console log, or a run's own log file, from a byte offset."""
    p = h.panel
    try:
        offset = int(q.get("offset", -1))
    except ValueError:
        raise HttpError(400, "bad offset") from None
    if "job" in q:
        job = p.jobs.get(q["job"])
        if job is None:
            raise HttpError(404, f"no job {q['job']}")
        return results.read_log(job.log_path, offset)
    path, kind = p.run_path(q.get("ref", ""))
    if kind != "train":
        return {"text": "", "offset": 0, "missing": True}
    return results.read_log(path / "train.log", offset)


def serve_file(h: Handler, q: dict) -> None:
    raw = q.get("path", "")
    path = resolve_path(raw) if raw else None
    if path is None or path.suffix.lower() not in SERVED_SUFFIXES or not h.panel.allowed(path):
        raise HttpError(403, "not a file the panel serves")
    if not path.is_file():
        raise HttpError(404, "file not found")
    h.send_file(path)


def picker(h: Handler, q: dict, kind: str) -> dict:
    return h.panel.picker(kind, q)


def upload(h: Handler, q: dict) -> dict:
    """Store a file chosen in the browser where a tool can read it (browsers hide local paths)."""
    kind, name = q.get("kind", ""), Path(q.get("name", "")).name
    if kind not in UPLOAD_SUFFIXES:
        raise HttpError(400, "kind must be json or video")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not safe or Path(safe).suffix.lower() not in UPLOAD_SUFFIXES[kind]:
        raise HttpError(400, f"expected a {' / '.join(UPLOAD_SUFFIXES[kind])} file")
    length = int(h.headers.get("Content-Length") or 0)
    if not 0 < length <= MAX_UPLOAD[kind]:
        raise HttpError(413, "empty or too large")
    folder = h.panel.folder / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}"
    remaining = length
    try:
        with target.open("wb") as f:
            while remaining:
                chunk = h.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    raise HttpError(400, "upload cut short")
                f.write(chunk)
                remaining -= len(chunk)
        if kind == "json":
            json.loads(target.read_text(encoding="utf-8"))
    except ValueError:
        target.unlink(missing_ok=True)
        raise HttpError(400, "not valid JSON") from None
    except BaseException:
        target.unlink(missing_ok=True)  # never leave a truncated file that looks valid
        raise
    return {"path": _rel(target), "name": name, "size": length}


def dataset_splits(h: Handler, q: dict) -> dict:
    return {"splits": h.panel.dataset.splits()}


def dataset_images(h: Handler, q: dict, split: str) -> dict:
    try:
        return h.panel.dataset.images(split)
    except (ValueError, FileNotFoundError) as e:
        raise HttpError(404, str(e)) from None


def dataset_points(h: Handler, q: dict, split: str) -> dict:
    try:
        return {"points": h.panel.dataset.points(split, q.get("name", ""))}
    except (ValueError, OSError) as e:
        raise HttpError(404, str(e)) from None


def label_studio(h: Handler, q: dict) -> dict:
    return h.panel.services.status(public=q.get("public") == "1")


def service_action(h: Handler, q: dict, name: str, action: str) -> dict:
    if name not in SERVICES:
        raise HttpError(404, f"unknown service {name!r}")
    body = h.json_body()
    services = h.panel.services
    try:
        return services.start(name, bool(body.get("public"))) if action == "start" else services.stop(name)
    except RuntimeError as e:
        raise HttpError(409, str(e)) from None


def service_log(h: Handler, q: dict, name: str) -> dict:
    if name not in SERVICES:
        raise HttpError(404, f"unknown service {name!r}")
    try:
        offset = int(q.get("offset", -1))
    except ValueError:
        raise HttpError(400, "bad offset") from None
    return results.read_log(h.panel.folder / "services" / f"{name}.log", offset)


ROUTES = {
    ("GET", "/"): page,
    ("GET", "/api/health"): health,
    ("POST", "/api/shutdown"): shutdown,
    ("POST", "/api/restart"): restart,
    ("GET", "/api/tools"): tools,
    ("POST", "/api/preview"): preview,
    ("POST", "/api/jobs"): start_job,
    ("GET", "/api/jobs"): list_jobs,
    ("POST", "/api/jobs/clear"): clear_jobs,
    ("GET", "/api/runs"): list_runs,
    ("GET", "/api/run"): run_detail,
    ("GET", "/api/log"): log,
    ("GET", "/api/file"): serve_file,
    ("POST", "/api/upload"): upload,
    ("GET", "/api/dataset"): dataset_splits,
    ("GET", "/api/label-studio"): label_studio,
}
PATTERN_ROUTES = [
    (("GET", re.compile(r"/static/([\w./-]+)")), static),
    (("GET", re.compile(r"/api/tools/([\w-]+)")), tool_spec),
    (("POST", re.compile(r"/api/jobs/([\w.-]+)/stop")), stop_job),
    (("GET", re.compile(r"/api/pickers/(\w+)")), picker),
    (("GET", re.compile(r"/api/dataset/([\w.-]+)")), dataset_images),
    (("GET", re.compile(r"/api/dataset/([\w.-]+)/points")), dataset_points),
    (("POST", re.compile(r"/api/services/(\w+)/(start|stop)")), service_action),
    (("GET", re.compile(r"/api/services/(\w+)/log")), service_log),
]


def make_server(panel: Panel, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("PanelHandler", (Handler,), {"panel": panel})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    panel.server = server
    return server


def serve(panel: Panel, host: str, port: int, on_ready=None) -> bool:
    """Serve until shut down; True when a restart was asked for."""
    server = make_server(panel, host, port)
    threading.Thread(target=_warm, args=(panel,), name="warm", daemon=True).start()
    if on_ready is not None:
        on_ready(server)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        server.server_close()
        panel.close()
    return panel.restart_requested.is_set()


def _warm(panel: Panel) -> None:
    """Read the tools' options now, off the request path: the first form must not wait."""
    try:
        panel.specs.load()
    except RuntimeError as e:
        print(f"[panel] {e}", file=sys.stderr, flush=True)
