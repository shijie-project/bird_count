"""FastAPI backend for the bird_count web UI.

Serves a single-page front end that drives train.py and test.py: it renders a
form from each script's argparse spec, launches the run as a child process,
polls its log, and shows the parsed metrics / evaluation results.

Run it with:  python -m webui           (then open http://127.0.0.1:8420)
"""

import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .runs import ROOT, RunManager
from .schema import build_argv, get_schema, list_entrypoints
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

# Label Studio, as launched by tools/starter.sh. The local port follows that
# script's LS_PORT; the public URL is the ngrok domain it forwards through.
LS_PORT = os.getenv("LS_PORT", "80")
LS_LOCAL_URL = os.getenv("LABEL_STUDIO_URL") or f"http://localhost:{LS_PORT}"
LS_PUBLIC_URL = os.getenv("LABEL_STUDIO_PUBLIC_URL", "https://obliging-maggot-frank.ngrok-free.app")
LS_PROBE_TIMEOUT = 2.5

# Files may only be served from inside the project's parent (…/code), which is
# where checkpoints, datasets and density-map overlays live.
FILE_ROOT = ROOT.parent.resolve()

manager = RunManager()
services = ServiceManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Never leave a training job or a service we started orphaned when the
    # server goes away — an abandoned Label Studio would keep holding its port.
    manager.stop_all()
    services.stop_all()


app = FastAPI(title="bird_count web UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


@app.get("/api/label-studio")
async def label_studio(public: bool = False) -> dict:
    """URLs for the annotation server, plus whether the selected one answers.

    The reachability probe runs server-side because the browser cannot check a
    cross-origin host, and a dead link is worth knowing about before clicking.
    """
    local_url = _normalize_url(LS_LOCAL_URL)
    public_url = _normalize_url(LS_PUBLIC_URL)
    url = public_url if public else local_url

    reachable, detail = False, "not configured"
    if url:
        try:
            async with httpx.AsyncClient(timeout=LS_PROBE_TIMEOUT, follow_redirects=True) as client:
                response = await client.get(url)
            # < 400 only: an idle ngrok domain answers 404 from ngrok itself,
            # which would otherwise look like a live Label Studio.
            reachable = response.status_code < 400
            detail = f"HTTP {response.status_code}"
        except Exception as exc:
            detail = type(exc).__name__

    # A tunnel started outside this UI (or left over from a hard kill) still
    # answers on ngrok's inspector port; report reality, not just what we own.
    tunnel_alive = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            tunnel_alive = (await client.get(NGROK_INSPECTOR)).status_code < 500
    except Exception:
        pass

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


@app.get("/api/label-studio/projects")
def label_studio_projects() -> dict:
    """Projects on the running server, for the --project-id picker."""
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    if not api_key:
        return {"items": [], "error": "LABEL_STUDIO_API_KEY is not set in .env"}
    try:
        from label_studio_sdk import LabelStudio

        client = LabelStudio(base_url=_normalize_url(LS_LOCAL_URL), api_key=api_key)
        items = [
            {
                "value": str(project.id),
                "label": f"{project.id} · {project.title}",
                "detail": f"{getattr(project, 'task_number', '?')} tasks",
            }
            for project in client.projects.list()
        ]
    except Exception as exc:
        return {"items": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"items": items}


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
def checkpoints(root: str = "../ckpts", limit: int = 300) -> dict:
    """List .pth/.tar checkpoints under `root`, newest first, for the ckpt picker."""
    try:
        base = _safe_path(root)
    except HTTPException:
        return {"root": root, "items": [], "error": "path outside the allowed directory"}
    if not base.is_dir():
        return {"root": str(base), "items": [], "error": "directory not found"}

    found = [p for pattern in ("*.pth", "*.tar") for p in base.rglob(pattern)]
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
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


def _require(run_id: str):
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    return run


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    return _require(run_id).detail()


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, cursor: int = 0) -> dict:
    """Incremental log tail. The client passes back the cursor it last received."""
    run = _require(run_id)
    lines, next_cursor = run.lines_since(cursor)
    payload = run.summary()
    payload.update({"lines": lines, "cursor": next_cursor, "metrics": run.metrics, "result": run.result})
    return payload


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    run = _require(run_id)
    run.stop()
    return run.summary()


@app.get("/api/runs/{run_id}/gallery")
def run_gallery(run_id: str, limit: int = 500) -> dict:
    """Density overlays written by test.py, newest run's output directory."""
    run = _require(run_id)
    raw = run.result.get("overlay_dir")
    if not raw:
        return {"dir": None, "items": []}
    directory = _safe_path(raw)
    if not directory.is_dir():
        return {"dir": str(directory), "items": []}

    # The overlay directory is shared by every run against the same checkpoint,
    # so restrict the gallery to the images this run actually evaluated.
    by_error = {img["name"]: img for img in run.result.get("images", [])}
    items = []
    for png in sorted(directory.glob("*_density.png")):
        name = png.name[: -len("_density.png")]
        if by_error and name not in by_error:
            continue
        info = by_error.get(name, {})
        items.append(
            {
                "name": name,
                "path": str(png),
                "gt": info.get("gt"),
                "pred": info.get("pred"),
                "err": info.get("err"),
                "rel": info.get("rel"),
            }
        )
    items.sort(key=lambda i: abs(i["err"]) if i["err"] is not None else -1, reverse=True)
    return {"dir": str(directory), "items": items[:limit]}


@app.get("/api/file")
def serve_file(path: str = Query(...)) -> FileResponse:
    resolved = _safe_path(path)
    if not resolved.is_file():
        raise HTTPException(404, "file not found")
    media_type, _ = mimetypes.guess_type(resolved.name)
    return FileResponse(resolved, media_type=media_type or "application/octet-stream")


def serve(host: str = "127.0.0.1", port: int = 8420, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run("webui.server:app" if reload else app, host=host, port=port, reload=reload)
