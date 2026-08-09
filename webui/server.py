"""FastAPI backend for the bird_count web UI.

Serves a single-page front end that drives train.py and test.py: it renders a
form from each script's argparse spec, launches the run as a child process,
polls its log, and shows the parsed metrics / evaluation results.

Run it with:  python -m webui           (then open http://127.0.0.1:8420)
"""

import asyncio
import json
import mimetypes
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import dotenv
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware

from .runs import ROOT, RunManager
from .schema import build_argv, get_schema, list_entrypoints, warm_cache
from .services import (
    LABEL_STUDIO,
    NGROK,
    NGROK_INSPECTOR,
    NGROK_INSPECTOR_PORT,
    ServiceManager,
    external_processes,
    kill_tree,
    label_studio_command,
    ngrok_command,
)


dotenv.load_dotenv(ROOT / ".env")

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = ROOT / "logs" / "webui" / "uploads"
MAX_JSON_UPLOAD = 100 * 1024 * 1024

# Label Studio, as launched by tools/starter.sh. The local port follows that
# script's LS_PORT; the public URL is the ngrok domain it forwards through.
LS_PORT = os.getenv("LS_PORT", "8080")
LS_LOCAL_URL = os.getenv("LABEL_STUDIO_URL") or f"http://localhost:{LS_PORT}"
LS_PUBLIC_URL = os.getenv("LABEL_STUDIO_PUBLIC_URL", "https://obliging-maggot-frank.ngrok-free.app")
LS_PROBE_TIMEOUT = 2.5
# A running inspector is a local Go server answering in single-digit ms; when it
# is absent this machine takes ~2s to refuse the connection, and that wait would
# be paid on every poll of the Label Studio page. Cap it well above the honest answer.
NGROK_PROBE_TIMEOUT = 0.3
# How long a project listing stays good. It costs a round trip to Label Studio
# and the form re-renders it on every tool switch.
PROJECTS_TTL = 60.0

# Files may only be served from inside the project's parent (…/code), which is
# where checkpoints, datasets and density-map overlays live.
FILE_ROOT = ROOT.parent.resolve()

manager = RunManager()
services = ServiceManager()


def _warm() -> None:
    """Pay the slow first-time costs up front, off the request path.

    Reading a script's argparse spec costs a subprocess that imports torch, and
    the first Label Studio call trades the API token for a session. Both are
    what made the first click on a tab sit there; neither needs a user waiting.
    """
    warm_cache()
    try:
        label_studio_projects()
    except Exception:  # Label Studio being down is not a startup problem
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm, name="webui-warmup", daemon=True).start()
    yield
    # Never leave a training job or a service we started orphaned when the
    # server goes away — an abandoned Label Studio would keep holding its port.
    manager.stop_all()
    services.stop_all()
    if _http is not None:
        await _http.aclose()


app = FastAPI(title="bird_count web UI", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def cache_versioned_assets(request, call_next):
    """Asset URLs carry an mtime version, so repeat visits can skip them."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") and request.url.query:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


class StartRequest(BaseModel):
    kind: str
    values: dict = {}


def _safe_path(raw: str) -> Path:
    """Resolve `raw` (absolute, or relative to the project root) inside FILE_ROOT."""
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(FILE_ROOT):
        raise HTTPException(400, f"path outside {FILE_ROOT}")
    return path


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the page with mtime-stamped asset URLs.

    Without this the browser happily keeps a cached style.css/app.js after an
    edit, which looks like the UI silently ignoring changes.
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for asset in ("style.css", "app.js"):
        version = int((STATIC_DIR / asset).stat().st_mtime)
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={version}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


def _normalize_url(url: str) -> str:
    """Accept a bare domain (as tools/starter.sh takes it) and drop a trailing slash."""
    url = url.strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


_http: Optional[httpx.AsyncClient] = None


def _client() -> httpx.AsyncClient:
    """One HTTP client for the whole app.

    Building an AsyncClient costs ~0.2s in transport and TLS setup — more than
    the probes it would carry. Two per poll made the Data page feel sticky.
    """
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(follow_redirects=True)
    return _http


async def _probe(url: str, timeout: float) -> tuple[Optional[int], str]:
    """Status code of a GET against `url`, or None plus the reason it failed."""
    if not url:
        return None, "not configured"
    try:
        response = await _client().get(url, timeout=timeout)
    except Exception as exc:
        return None, type(exc).__name__
    return response.status_code, f"HTTP {response.status_code}"


@app.get("/api/label-studio")
async def label_studio(public: bool = False) -> dict:
    """URLs for the annotation server, plus whether the selected one answers.

    The reachability probe runs server-side because the browser cannot check a
    cross-origin host, and a dead link is worth knowing about before clicking.
    """
    local_url = _normalize_url(LS_LOCAL_URL)
    public_url = _normalize_url(LS_PUBLIC_URL)
    url = public_url if public else local_url

    # Both probes are network round trips and neither depends on the other, so
    # they go together — one after the other doubles the cost of every poll.
    # A tunnel started outside this UI (or left over from a hard kill) still
    # answers on ngrok's inspector port; report reality, not just what we own.
    (status, detail), (tunnel_status, _) = await asyncio.gather(
        _probe(url, LS_PROBE_TIMEOUT),
        _probe(NGROK_INSPECTOR, NGROK_PROBE_TIMEOUT),
    )
    # < 400 only: an idle ngrok domain answers 404 from ngrok itself, which
    # would otherwise look like a live Label Studio.
    reachable = status is not None and status < 400
    tunnel_alive = tunnel_status is not None and tunnel_status < 500

    domain = public_url.removeprefix("https://")
    return {
        "ngrok_alive": tunnel_alive,
        "local_url": local_url,
        "public_url": public_url,
        "url": url,
        "reachable": reachable,
        "detail": detail,
        "start_command": "./tools/starter.sh label-studio" + (f" {domain}" if public and public_url else ""),
        "ngrok_command": f"ngrok http --url={domain} {LS_PORT}" if public_url else "",
        "ngrok_inspector": NGROK_INSPECTOR,
        "service": services.status(LABEL_STUDIO),
        "services": {name: services.status(name) for name in (LABEL_STUDIO, NGROK)},
    }


class ServiceStart(BaseModel):
    public: bool = False


def _service_command(name: str, public: bool) -> tuple[list[str], dict]:
    domain = _normalize_url(LS_PUBLIC_URL)
    if name == LABEL_STUDIO:
        return label_studio_command(domain if public else "")
    if name == NGROK:
        return ngrok_command(domain, LS_PORT)
    raise HTTPException(404, f"unknown service '{name}'")


@app.post("/api/services/{name}/start")
def service_start(name: str, req: ServiceStart) -> dict:
    """Launch a background server the same way tools/starter.sh would."""
    try:
        argv, env = _service_command(name, req.public)
    except RuntimeError as exc:  # missing binary or missing public domain
        raise HTTPException(503, str(exc))
    try:
        services.start(name, argv, env, values={"public": req.public})
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return services.status(name)


def _service_port(name: str) -> str:
    return NGROK_INSPECTOR_PORT if name == NGROK else LS_PORT


@app.post("/api/services/{name}/stop")
def service_stop(name: str) -> dict:
    """Stop the service — ours if we started it, otherwise the one on its port.

    Adopting an externally started server is deliberate: it is the same server
    the page is reporting on, and leaving Stop greyed out just forces the user
    into a terminal. `external_processes` refuses anything whose command line is
    not that service, so an unrelated listener on the port is never touched.
    """
    if name not in (LABEL_STUDIO, NGROK):
        raise HTTPException(404, f"unknown service '{name}'")

    service = services.get(name)
    if service is not None and service.running:
        services.stop(name)
        status = services.status(name)
        status.update({"stopped": "owned", "killed": [], "refused": []})
        return status

    killed, refused = [], []
    for process in external_processes(name, _service_port(name)):
        if not process["matches"]:
            refused.append(process)
        elif kill_tree(process["pid"]):
            killed.append(process)

    status = services.status(name)
    status.update({"stopped": "external" if killed else "nothing", "killed": killed, "refused": refused})
    return status


@app.get("/api/services/{name}/log")
def service_log(name: str, cursor: int = 0) -> dict:
    """Incremental tail of a service log, same cursor protocol as runs."""
    service = services.get(name)
    if service is None:
        return {"lines": [], "cursor": 0, "state": "stopped", "started": False}
    lines, next_cursor = service.lines_since(cursor)
    payload = services.status(name)
    payload.update({"lines": lines, "cursor": next_cursor})
    return payload


_ls_clients: dict[tuple[str, str], object] = {}
_projects_cache: dict = {"key": None, "at": 0.0, "data": None}


def _ls_client(url: str, api_key: str):
    """One SDK client per target, reused.

    Constructing it trades the personal access token for a session, which costs
    around two seconds; on a client that already did that, listing projects is
    ~50ms. Building a fresh one per request paid the toll every single time.
    """
    cache_key = (url, api_key)
    client = _ls_clients.get(cache_key)
    if client is None:
        from label_studio_sdk import LabelStudio

        client = LabelStudio(base_url=url, api_key=api_key)
        _ls_clients[cache_key] = client
    return client


@app.get("/api/label-studio/projects")
def label_studio_projects(refresh: bool = False) -> dict:
    """Projects on the running server, for the --project-id picker."""
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    if not api_key:
        return {"items": [], "error": "LABEL_STUDIO_API_KEY is not set in .env"}

    url = _normalize_url(LS_LOCAL_URL)
    cache_key = (url, api_key)
    now = time.monotonic()
    if not refresh and _projects_cache["key"] == cache_key and now - _projects_cache["at"] < PROJECTS_TTL:
        return _projects_cache["data"]

    try:
        items = [
            {
                "value": str(project.id),
                "label": f"{project.id} · {project.title}",
                "detail": f"{getattr(project, 'task_number', '?')} tasks",
            }
            for project in _ls_client(url, api_key).projects.list()
        ]
    except Exception as exc:
        _ls_clients.pop(cache_key, None)  # an expired session must not be reused
        return {"items": [], "error": f"{type(exc).__name__}: {exc}"}

    data = {"items": items}
    _projects_cache.update(key=cache_key, at=now, data=data)  # only successes are cached
    return data


# The running uvicorn Server, set by serve() so /api/shutdown can ask it to stop.
_server = None


@app.post("/api/shutdown")
def shutdown() -> dict:
    """Stop the web UI itself and release its port.

    The reply goes out first and the process follows a moment later, so the page
    can report what happened instead of dying on a failed request. Everything
    this server started goes down with it — the active run, Label Studio, the
    tunnel — rather than being left holding a port or a GPU.
    """
    active = manager.active()
    running = [name for name in (LABEL_STUDIO, NGROK) if (s := services.get(name)) is not None and s.running]

    def stop() -> None:
        time.sleep(0.4)  # long enough for the response to reach the browser
        if _server is not None:
            _server.should_exit = True  # graceful: the lifespan stops the children
            return
        # Started with --reload, so the reloader owns the process and there is no
        # Server object to ask nicely; do its cleanup by hand and go.
        manager.stop_all()
        services.stop_all()
        os._exit(0)

    threading.Thread(target=stop, name="shutdown", daemon=True).start()
    return {"stopping": True, "run": active.id if active else None, "services": running}


@app.get("/api/entrypoints")
def entrypoints() -> dict:
    return {"entrypoints": list_entrypoints()}


@app.get("/api/schema/{kind}")
def schema(kind: str) -> dict:
    try:
        return get_schema(kind)
    except KeyError:
        raise HTTPException(404, f"unknown entrypoint '{kind}'")
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/checkpoints")
def checkpoints(root: str = "../ckpts", limit: int = 300, match: str = "") -> dict:
    """List checkpoints under `root`, newest first, for the ckpt picker.

    `match` is a comma-separated list of file-name globs replacing the default
    .pth/.tar sweep. ../ckpts holds three shapes of checkpoint: one `best.pth`
    per run directory, the `best_ep<N>_mae...pth` files promoted to the top, and
    the periodic `<epoch>_ckpt.tar` dumps (plus `best_model_<N>.pth` from the old
    trainer, kept under ckpts_legacy/). Evaluation wants the first two, so the
    picker names them instead of walking the tree for everything.
    """
    try:
        base = _safe_path(root)
    except HTTPException:
        return {"root": root, "items": [], "error": "path outside the allowed directory"}
    if not base.is_dir():
        return {"root": str(base), "items": [], "error": "directory not found"}

    patterns = [g.strip() for g in match.split(",") if g.strip()] or ["*.pth", "*.tar"]
    # A set: overlapping globs must not list the same file twice.
    found = sorted({p for g in patterns for p in base.rglob(g)}, key=lambda p: p.stat().st_mtime, reverse=True)
    items = [
        {
            # Relative to the project root so the generated command line stays
            # portable (checkpoints usually live in a sibling ../ckpts).
            "path": os.path.relpath(p, ROOT).replace(os.sep, "/"),
            "label": str(p.relative_to(base)),
            "mtime": p.stat().st_mtime,
            "size_mb": round(p.stat().st_size / 1e6, 1),
        }
        for p in found[:limit]
    ]
    return {"root": str(base), "items": items}


@app.post("/api/uploads/json")
async def upload_json(request: Request, filename: str = Query(...)) -> dict:
    """Store a browser-selected JSON export where a WebUI op can read it.

    Browsers intentionally do not reveal a selected file's real local path, so
    the generic form uploads the content and receives a project-relative path
    suitable for the generated command line.
    """
    original = Path(filename).name
    if Path(original).suffix.lower() != ".json":
        raise HTTPException(400, "only .json files can be uploaded")
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > MAX_JSON_UPLOAD:
                raise HTTPException(413, "JSON upload exceeds 100 MB")
        except ValueError:
            raise HTTPException(400, "invalid Content-Length header") from None
    payload = await request.body()
    if len(payload) > MAX_JSON_UPLOAD:
        raise HTTPException(413, "JSON upload exceeds 100 MB")
    try:
        json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"invalid JSON: {exc}") from exc

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._") or "export.json"
    if not safe_name.lower().endswith(".json"):
        safe_name += ".json"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{time.time_ns()}-{safe_name}"
    target.write_bytes(payload)
    return {
        "path": os.path.relpath(target, ROOT).replace(os.sep, "/"),
        "name": original,
        "size": len(payload),
    }


@app.post("/api/runs")
def start_run(req: StartRequest) -> dict:
    try:
        argv, env = build_argv(req.kind, req.values)
    except KeyError:
        raise HTTPException(404, f"unknown entrypoint '{req.kind}'")
    except ValueError as exc:  # a required field was left empty
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    try:
        run = manager.start(req.kind, argv, req.values, env=env)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return run.detail()


@app.get("/api/runs")
def list_runs() -> dict:
    active = manager.active()
    return {"runs": manager.list(), "active": active.id if active else None}


@app.post("/api/runs/clear")
def clear_runs() -> dict:
    """Wipe the run history: drop finished runs and delete their log files.

    History is restored from logs/webui/ at startup, so forgetting a run only
    sticks if its log goes with it. A run still in flight is never touched.
    """
    removed, failed = manager.clear()
    active = manager.active()
    return {"removed": len(removed), "failed": failed, "kept": active.id if active else None}


def _require(run_id: str):
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    return run


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    return _require(run_id).detail()


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, cursor: int = 0, tail: int = 0, state_version: Optional[int] = None) -> dict:
    """Incremental log tail. The client passes back the cursor it last received."""
    run = _require(run_id)
    lines, next_cursor = run.lines_since(cursor, min(max(tail, 0), 5000))
    payload = run.summary()
    payload.update({"lines": lines, "cursor": next_cursor, "state_version": run.state_version})
    if state_version != run.state_version:
        payload.update({"metrics": run.metrics, "result": run.result})
    return payload


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    run = _require(run_id)
    run.stop()
    return run.summary()


@app.get("/api/runs/{run_id}/gallery")
def run_gallery(run_id: str, limit: int = 500) -> dict:
    """Overlay PNGs written by the run, from the output directory it reported.

    Two tools produce a gallery: test.py writes `<name>_density.png` and reports
    a GT/pred error per image, density_regions.py writes `<name>_regions.png`
    and reports a count per image. The per-image record is passed through as-is
    and the client decides how to caption it.
    """
    run = _require(run_id)
    raw = run.result.get("overlay_dir")
    if not raw:
        return {"dir": None, "items": []}
    directory = _safe_path(raw)
    if not directory.is_dir():
        return {"dir": str(directory), "items": []}

    suffix = run.result.get("overlay_suffix") or "_density.png"
    # An overlay directory is shared by every run that targeted it, so restrict
    # the gallery to the images this run actually processed. test.py logs the
    # image stem, density_regions.py logs the file name, and the PNG is named
    # after the stem — so index by both and look up on the stem.
    by_name: dict[str, dict] = {}
    for img in run.result.get("images", []):
        by_name[img["name"]] = img
        by_name.setdefault(Path(img["name"]).stem, img)

    items = []
    for png in sorted(directory.glob(f"*{suffix}")):
        name = png.name[: -len(suffix)]
        if by_name and name not in by_name:
            continue
        items.append({**by_name.get(name, {}), "name": name, "path": str(png)})

    def natural_name(item: dict) -> tuple:
        """Case-insensitive filename order with numeric chunks sorted as numbers."""
        parts = re.split(r"(\d+)", str(item.get("name", "")))
        return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts)

    def rank(item: dict) -> float:
        """Worst first for an evaluation, busiest first for regions."""
        if item.get("worst_blob") is not None:
            return item["worst_blob"]
        if item.get("worst_region") is not None:
            return item["worst_region"]
        if item.get("err") is not None:
            return abs(item["err"])
        return item.get("total") or 0.0

    if run.kind == "density_regions":
        items.sort(key=natural_name)
        order = "name"
    else:
        items.sort(key=rank, reverse=True)
        order = "worst"
    return {"dir": str(directory), "items": items[:limit], "order": order}


@app.get("/api/file")
def serve_file(path: str = Query(...)) -> FileResponse:
    resolved = _safe_path(path)
    if not resolved.is_file():
        raise HTTPException(404, "file not found")
    media_type, _ = mimetypes.guess_type(resolved.name)
    return FileResponse(resolved, media_type=media_type or "application/octet-stream")


def serve(host: str = "127.0.0.1", port: int = 8420, reload: bool = False) -> None:
    import uvicorn

    if reload:  # the reloader owns the process; /api/shutdown falls back to exit
        uvicorn.run("webui.server:app", host=host, port=port, reload=True)
        return

    global _server
    # Built explicitly instead of through uvicorn.run() so that /api/shutdown has
    # a Server to ask for a graceful stop.
    _server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
    _server.run()
