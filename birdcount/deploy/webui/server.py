"""The farm backend's HTTP server: the page and its JSON API over the runtime's files.

Read-only except for labels: the only thing a request can write is a verdict
on a snapshot, into that day's labels.csv. Images are served from the count
log's snapshot folders and the alarm evidence folder, nothing else.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from birdcount.core.camera_identity import index_by_mac
from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.core.webserver import HttpError, LocalHandler
from birdcount.deploy import countlog
from birdcount.deploy.alarm import AppConfig
from birdcount.deploy.calibrate.rubric import LEVELS, RUBRIC_VERSION, SHADE
from birdcount.deploy.topology import Topology
from birdcount.deploy.webui import store


logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
KINDS = ("high", "bin", "alarm")


class Backend:
    """Everything the request handlers share."""

    def __init__(
        self,
        config: AppConfig,
        topology: Topology,
        *,
        status_path: Path | None,
        count_log_dir: Path | None,
        evidence_dir: Path,
        calibration_dir: Path | None = None,
        calibration_model: str | None = None,
        mask_dir: Path | None = None,
        recovery_ratio: float | None = None,
    ):
        self.config = config
        self.cameras = store.cameras_of(config, topology)
        self.by_mac = {c.mac: c for c in self.cameras}
        self.zones = store.zones_of(topology, self.cameras)
        self.status_path = status_path
        self.logs = store.CountLogs(count_log_dir)
        self.labels = store.Labels()
        self.evidence = store.Evidence(evidence_dir)
        self.calibration_dir = calibration_dir
        self.calibration_model = calibration_model
        self.masks: dict[str, Path] = {}
        if mask_dir is not None and mask_dir.is_dir():
            self.masks, _ = index_by_mac(p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.recovery_ratio = config.rules.recovery_ratio if recovery_ratio is None else recovery_ratio
        self.server: ThreadingHTTPServer | None = None

    def camera(self, mac: str) -> store.CameraInfo:
        cam = self.by_mac.get(mac.upper())
        if cam is None:
            raise HttpError(404, f"no camera {mac}")
        return cam

    def day_dir(self, day: str) -> Path:
        try:
            return self.logs.day_dir(day)
        except KeyError:
            raise HttpError(404, f"no count log for {day}") from None

    def ratings(self, day_dir: Path) -> dict[str, dict]:
        """The vision model's crowding level per picture of the day (the configured model only)."""
        from birdcount.deploy.calibrate.ratings import current, read

        return {r["snapshot"]: r for r in current(read(day_dir), self.calibration_model or "")}

    def pictures(self, day: str, cam: store.CameraInfo, actions: list[dict]) -> list[dict]:
        """A camera's pictures of the day, keyed like its labels: count-log snapshots, then alarm pictures."""
        s = self.logs.series(day, cam.mac)
        threshold = self.threshold_on(day, cam)
        out = []
        for t, rel in s.snapshots if s else ():
            info = countlog.parse_snapshot(day, rel) or {"path": rel, "kind": "", "count": 0}
            out.append(
                {
                    **info,
                    "key": rel,
                    "source": "count_log",
                    "t": t,
                    "mac": cam.mac,
                    "threshold": threshold,
                    "level": _level_at(s, t),
                }
            )
        out.extend(
            {
                "key": f"alarm:{a['image']}",
                "path": a["image"],
                "source": "alarm",
                "t": int(a["timestamp"]),
                "kind": a["action_type"],
                "count": round(a["count"]),
                "mac": cam.mac,
                "threshold": a["threshold"],
                "level": a.get("level") or 0,
            }
            for a in actions
            if a["mac"] == cam.mac and a.get("image")
        )
        return out

    def threshold_on(self, day: str, cam: store.CameraInfo) -> float:
        """The threshold in force that day (meta.json), else today's."""
        return float(self.logs.meta(day).get("thresholds", {}).get(cam.mac, cam.threshold))


def _camera_json(cam: store.CameraInfo) -> dict:
    return {
        "mac": cam.mac,
        "camera_id": cam.camera_id,
        "name": cam.name,
        "threshold": cam.threshold,
        "zone": cam.zone,
        "group": cam.group,
    }


def _action_json(a: dict) -> dict:
    return {
        "type": a.get("action_type"),
        "mac": a.get("mac"),
        "event_id": a.get("event_id"),
        "t": a.get("timestamp"),
        "count": a.get("count"),
        "threshold": a.get("threshold"),
        "level": a.get("level"),
        "image": a.get("image"),
    }


# --- routes ------------------------------------------------------------------------------------------


class Handler(LocalHandler):
    server_version = "birdcount-farm"
    backend: Backend  # set on the subclass made by make_server


def page(h: Handler, q: dict) -> None:
    h.send_file(STATIC_DIR / "index.html")


def static(h: Handler, q: dict, name: str) -> None:
    h.send_static(STATIC_DIR, name)


def farm(h: Handler, q: dict) -> dict:
    b = h.backend
    return {
        "cameras": [_camera_json(c) for c in b.cameras],
        "zones": b.zones,
        "placed": all(c.zone for c in b.cameras),
        "status": store.read_status(b.status_path),
        "days": b.logs.days(),
        "recovery_ratio": b.recovery_ratio,
        "masks": sorted(b.masks),
        "mask_shade": SHADE,
    }


def status(h: Handler, q: dict) -> dict:
    return store.read_status(h.backend.status_path)


def day_summary(h: Handler, q: dict, day: str) -> dict:
    b = h.backend
    day_dir = b.day_dir(day)
    labels = b.labels.read(day_dir)
    levels = {key: int(r["level"]) for key, r in b.ratings(day_dir).items()}
    evidence = b.evidence.of_day(day)
    meta = b.logs.meta(day)
    cameras = {}
    for cam in b.cameras:
        threshold = b.threshold_on(day, cam)
        summary = store.summarize(b.logs.series(day, cam.mac), threshold)
        pictures = b.pictures(day, cam, evidence)
        queue = store.label_queue(pictures, threshold, levels, set(labels))
        summary["threshold"] = threshold
        summary["queued"] = len(queue)
        summary["queue_done"] = sum(1 for key in queue if key in labels)
        summary["to_label"] = sorted(p["t"] for p in pictures if p["key"] in queue and p["key"] not in labels)
        cameras[cam.mac] = summary
    actions = [_action_json(a) for a in evidence]
    spans = [(c["first"], c["last"]) for c in cameras.values() if c.get("seconds")]
    return {
        "day": day,
        "hours": meta.get("hours"),
        "model": Path(meta.get("model_path") or "").name or None,
        "span": [min(a for a, _ in spans), max(z for _, z in spans)] if spans else None,
        "cameras": cameras,
        "actions": actions,
    }


def camera_day(h: Handler, q: dict, day: str, mac: str) -> dict:
    b = h.backend
    cam = b.camera(mac)
    day_dir = b.day_dir(day)
    s = b.logs.series(day, cam.mac)
    threshold = b.threshold_on(day, cam)
    labels = b.labels.read(day_dir)
    rated = b.ratings(day_dir)
    snaps = []
    for t, rel in s.snapshots if s else ():
        info = countlog.parse_snapshot(day, rel) or {"path": rel, "t": t, "kind": "", "count": None}
        info["t"] = t
        info["level"] = _level_at(s, t)
        info["label"] = labels.get(rel, {}).get("label")
        info["rating"] = _rating(rated.get(rel))
        snaps.append(info)
    return {
        "day": day,
        "camera": _camera_json(cam),
        "threshold": threshold,
        "clear_level": round(threshold * b.recovery_ratio, 1),
        "summary": store.summarize(s, threshold),
        "curve": store.curve(s, start=_number(q, "from"), end=_number(q, "to")) if s else None,
        "snapshots": snaps,
        "actions": [_action_json(a) for a in b.evidence.of_day(day) if a["mac"] == cam.mac],
    }


def snapshots(h: Handler, q: dict) -> dict:
    """Pictures to label. state=todo (the default): what is left of the day's queue, a few per
    camera (store.label_queue); done / all: every picture of the kinds asked, labelled or all."""
    b = h.backend
    day = q.get("day") or next(iter(b.logs.days()), "")
    day_dir = b.day_dir(day)
    labels = b.labels.read(day_dir)
    rated = b.ratings(day_dir)
    levels = {key: int(r["level"]) for key, r in rated.items()}
    evidence = b.evidence.of_day(day)
    kinds = set((q.get("kind") or "high,bin,alarm").split(","))
    state = q.get("state", "todo")
    cams = [b.camera(q["camera"])] if q.get("camera") else b.cameras
    items, done, total = [], 0, 0
    for cam in cams:
        pictures = b.pictures(day, cam, evidence)
        queue = store.label_queue(pictures, b.threshold_on(day, cam), levels, set(labels))
        for p in pictures:
            if ("alarm" if p["source"] == "alarm" else p["kind"]) not in kinds:
                continue
            if state == "todo" and p["key"] not in queue:
                continue
            label = labels.get(p["key"], {}).get("label")
            total += 1
            done += label is not None
            if (state == "todo" and label) or (state == "done" and not label):
                continue
            item = {k: v for k, v in p.items() if k != "key"}
            items.append({**item, "label": label, "rating": _rating(rated.get(p["key"])), "why": queue.get(p["key"])})
    items.sort(key=lambda i: (i["t"], i["mac"]))
    return {"day": day, "items": items, "labelled": done, "total": total}


def save_label(h: Handler, q: dict) -> dict:
    """{day, path, source: "count_log" | "alarm", label}: one verdict into that day's labels.csv."""
    body = h.json_body()
    b = h.backend
    day_dir = b.day_dir(str(body.get("day", "")))
    rel = str(body.get("path", ""))
    alarm = body.get("source") == "alarm"
    root = b.evidence.root if alarm else day_dir
    path = (root / rel).resolve()
    if not rel or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise HttpError(400, f"no snapshot {rel!r} to label")
    try:
        return b.labels.write(day_dir, f"alarm:{rel}" if alarm else rel, str(body.get("label") or ""))
    except ValueError as e:
        raise HttpError(400, str(e)) from None


def image(h: Handler, q: dict) -> None:
    """?day=<day>&path=<snapshot path> from a count log, or ?alarm=<path> under the evidence folder."""
    b = h.backend
    if "alarm" in q:
        root, rel = b.evidence.root, q["alarm"]
    else:
        root, rel = b.day_dir(q.get("day", "")), q.get("path", "")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HttpError(403, "not an image this backend serves")
    if not path.is_file():
        raise HttpError(404, "image not found")
    h.send_file(path, cache="max-age=86400")  # a snapshot never changes once written


def ratings(h: Handler, q: dict) -> dict:
    """One day's pictures as the vision model rated them, by camera: count, crowding level, reason."""
    b = h.backend
    day = q.get("day") or next(iter(b.logs.days()), "")
    day_dir = b.day_dir(day)
    rated = b.ratings(day_dir)
    cameras = {}
    for cam in b.cameras:
        s = b.logs.series(day, cam.mac)
        snaps = s.snapshots if s else ()
        items = [_rated(rel, "count_log", t, rated[rel]) for t, rel in snaps if rel in rated]
        cameras[cam.mac] = {"threshold": b.threshold_on(day, cam), "pictures": len(snaps), "items": items}
    for a in b.evidence.of_day(day):
        cam = cameras.get(a["mac"])
        if cam is None or not a.get("image"):
            continue
        cam["pictures"] += 1
        r = rated.get(f"alarm:{a['image']}")
        if r:
            cam["items"].append(_rated(a["image"], "alarm", int(a["timestamp"]), r))
    return {
        "day": day,
        "model": b.calibration_model,
        "rubric": RUBRIC_VERSION,
        "levels": {n: {"name": text.split(":")[0], "text": text} for n, text in LEVELS.items()},
        "cameras": cameras,
    }


def _rated(path: str, source: str, t: int, r: dict) -> dict:
    return {
        "path": path,
        "source": source,
        "t": t,
        "count": round(float(r["count"])),
        "kind": r.get("kind", ""),
        "level": int(r["level"]),
        "reason": r.get("reason", ""),
    }


def calibration(h: Handler, q: dict) -> dict:
    """The latest threshold proposals (deploy.calibrate propose), with the thresholds in force now."""
    b = h.backend
    path = b.calibration_dir / "latest.json" if b.calibration_dir else None
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path else None
    except (OSError, ValueError):
        data = None
    return {"proposal": data, "model": b.calibration_model, "thresholds": {c.mac: c.threshold for c in b.cameras}}


def mask(h: Handler, q: dict, mac: str) -> None:
    """A camera's region mask (white = counted), for the page to shade the rest of a picture."""
    path = h.backend.masks.get(mac.upper())
    if path is None:
        raise HttpError(404, f"no mask for {mac}")
    h.send_file(path, cache="no-cache")


def _number(q: dict, name: str) -> float | None:
    try:
        return float(q[name]) if q.get(name) else None
    except ValueError:
        raise HttpError(400, f"{name} must be a number") from None


def _rating(r: dict | None) -> dict | None:
    return {"level": int(r["level"]), "reason": r.get("reason", "")} if r else None


def _level_at(s, t: int) -> int:
    i = bisect.bisect_left(s.ts, t)
    return int(s.level[i]) if i < len(s.ts) and s.ts[i] == t else 0


Handler.routes = {
    ("GET", "/"): page,
    ("GET", "/api/farm"): farm,
    ("GET", "/api/status"): status,
    ("GET", "/api/snapshots"): snapshots,
    ("POST", "/api/labels"): save_label,
    ("GET", "/api/image"): image,
    ("GET", "/api/calibration"): calibration,
    ("GET", "/api/ratings"): ratings,
}
Handler.pattern_routes = [
    (("GET", re.compile(r"/static/([\w./-]+)")), static),
    (("GET", re.compile(r"/api/day/(\d{4}-\d{2}-\d{2})")), day_summary),
    (("GET", re.compile(r"/api/mask/([0-9A-Fa-f]{12})")), mask),
    (("GET", re.compile(r"/api/day/(\d{4}-\d{2}-\d{2})/camera/([0-9A-Fa-f]{12})")), camera_day),
]


def make_server(backend: Backend, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("FarmHandler", (Handler,), {"backend": backend})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    backend.server = server
    return server


def serve(backend: Backend, host: str, port: int, on_ready=None) -> None:
    server = make_server(backend, host, port)
    threading.Thread(target=_warm, args=(backend,), name="warm", daemon=True).start()
    if on_ready is not None:
        on_ready(server)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        server.server_close()


def _warm(backend: Backend) -> None:
    """Read the latest day's count logs now, so the first page does not wait for them."""
    days = backend.logs.days()
    if days:
        for cam in backend.cameras:
            backend.logs.series(days[0], cam.mac)
        backend.evidence.actions()
