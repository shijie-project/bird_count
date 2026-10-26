"""What the farm backend reads: the runtime's own files, never its memory.

    status.json              the live state (handlers/status.py), every second
    count_log/<day>/         per camera per second + sampled snapshots (handlers/count_logger.py)
        labels.csv           written here: a person's verdict on a snapshot
    <alarm evidence>/        events.jsonl and events/<camera>/<event>/ (alarm/evidence.py)

Everything is read on request and cached by file size and mtime, so the
backend can start, stop and restart independently of the runtime, and shows
history while the runtime is down. A count log grows by appending, so the
file of the current day is read incrementally from where the last read
stopped instead of from the top every few seconds.

stdlib only.
"""

from __future__ import annotations

import csv
import io
import json
import re
import threading
import time
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.deploy import countlog
from birdcount.deploy.alarm import AppConfig
from birdcount.deploy.alarm.camera_ids import mac_of
from birdcount.deploy.calibrate.rubric import DENSE_LEVEL
from birdcount.deploy.topology import Topology


LABELS = ("pileup", "normal", "unsure")
LABEL_FIELDS = ["snapshot", "label", "labeled_at"]
STALE_AFTER_SEC = 10.0  # status.json not rewritten for this long: the runtime is not running
MAX_POINTS = 1500  # points per camera curve; a 14-hour day at 1 Hz is ~50 000 rows
QUEUE_PER_CAMERA = 3  # pictures a person is asked to label per camera and day
QUIET_LEVEL = 3  # a rating at or below this, at a count over the threshold, is a disagreement
_ZONE_SIDE = re.compile(r"(left|right)", re.I)
_ZONE_ROW = re.compile(r"(front|back|outer)", re.I)


# --- the farm --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraInfo:
    mac: str
    camera_id: str  # the alarm config's `axisN/MAC`
    name: str
    threshold: float
    zone: str | None  # topology zone; None when the MAC is not in topology.yaml
    group: str  # where the map draws it: the zone, else the recorder group `axisN`


def cameras_of(config: AppConfig, topology: Topology) -> list[CameraInfo]:
    """Every camera the alarm covers, placed in its zone when topology.yaml knows its MAC."""
    out = []
    for camera_id, cam in config.cameras.items():
        mac = mac_of(camera_id)
        zone = topology.zone_of_mac(mac)
        group = zone.name if zone else camera_id.split("/", 1)[0] if "/" in camera_id else "unplaced"
        out.append(CameraInfo(mac, camera_id, cam.name, cam.threshold, zone.name if zone else None, group))
    return out


def zones_of(topology: Topology, cameras: list[CameraInfo]) -> list[dict]:
    """The map's blocks: topology zones in floor-plan position, then any recorder groups.

    A zone named like `Zone_01_Right_Front` lands on its side and row of the
    shed; `Outer` zones go to the side column. Cameras whose MAC topology.yaml
    lacks are grouped by their recorder (`axisN`), which the map says plainly
    instead of guessing a zone.
    """
    blocks = []
    for z in topology.zones:
        side = (_ZONE_SIDE.search(z.name) or [None, None])[1]
        row = (_ZONE_ROW.search(z.name) or [None, None])[1]
        blocks.append(
            {
                "name": z.name,
                "kind": "zone",
                "side": side.lower() if side else None,
                "row": row.lower() if row else None,
                "cameras_listed": len(z.cameras),
                "speakers": len(z.speakers),
                "smart_plugs": len(z.smart_plugs),
                "macs": [c.mac for c in cameras if c.zone == z.name],
            }
        )
    groups = sorted({c.group for c in cameras if c.zone is None})
    blocks += [{"name": g, "kind": "group", "macs": [c.mac for c in cameras if c.group == g]} for g in groups]
    return blocks


# --- live -------------------------------------------------------------------------------------------


def read_status(path: Path | None, now: float | None = None) -> dict:
    """The runtime's status file and what it means right now: running, stopped or gone."""
    now = time.time() if now is None else now
    if path is None:
        return {"state": "disabled"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"state": "never"}
    except (OSError, ValueError):
        return {"state": "unreadable"}  # mid-replace on Windows: the next poll reads it
    age = now - float(data.get("written") or 0)
    state = "stopped" if data.get("stopped") else ("running" if age <= STALE_AFTER_SEC else "gone")
    return {"state": state, "age": age, **data}


# --- count logs --------------------------------------------------------------------------------------


@dataclass
class _Series:
    """One camera's day, column by column: compact enough for 20 cameras x 50 000 seconds."""

    ts: array = field(default_factory=lambda: array("l"))
    mean: array = field(default_factory=lambda: array("f"))
    low: array = field(default_factory=lambda: array("f"))
    high: array = field(default_factory=lambda: array("f"))
    level: array = field(default_factory=lambda: array("b"))
    excluded: array = field(default_factory=lambda: array("l"))
    snapshots: list[tuple[int, str]] = field(default_factory=list)  # (second, path relative to the day)
    offset: int = 0  # bytes consumed
    header: list[str] | None = None
    size: int = 0
    mtime: float = 0.0


class CountLogs:
    def __init__(self, root: Path | None):
        self.root = root
        self._series: dict[Path, _Series] = {}
        self._lock = threading.Lock()

    def days(self) -> list[str]:
        return countlog.days(self.root)

    def day_dir(self, day: str) -> Path:
        if self.root is None or not countlog.DAY_RE.fullmatch(day) or not (self.root / day).is_dir():
            raise KeyError(day)
        return self.root / day

    def meta(self, day: str) -> dict:
        try:
            return json.loads((self.day_dir(day) / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def series(self, day: str, mac: str) -> _Series | None:
        path = self.day_dir(day) / f"{mac}.csv"
        try:
            st = path.stat()
        except OSError:
            return None
        with self._lock:
            s = self._series.get(path)
            if s is None or st.st_size < s.size:  # new, or rewritten shorter
                s = self._series[path] = _Series()
            if st.st_size != s.size or st.st_mtime != s.mtime:
                self._read_more(path, s)
                s.size, s.mtime = st.st_size, st.st_mtime
            return s

    @staticmethod
    def _read_more(path: Path, s: _Series) -> None:
        with path.open("rb") as f:
            f.seek(s.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1  # a row still being written stays for the next read
        if not end:
            return
        s.offset += end
        lines = io.StringIO(data[:end].decode("utf-8"), newline="")
        reader = csv.reader(lines)
        if s.header is None:
            s.header = next(reader, None)
            if s.header is None:
                return
        col = {name: i for i, name in enumerate(s.header)}
        for row in reader:
            try:
                t = int(row[col["timestamp"]])
                s.ts.append(t)
                s.mean.append(float(row[col["count_mean"]]))
                s.low.append(float(row[col["count_min"]]))
                s.high.append(float(row[col["count_max"]]))
                s.level.append(int(row[col["alarm_level"]]))
                s.excluded.append(int(row[col["motion_excluded_frames"]] or 0))
            except (IndexError, KeyError, ValueError):
                continue
            snap = row[col["snapshot"]] if len(row) > col.get("snapshot", len(row)) else ""
            if snap:
                s.snapshots.append((t, snap))


def summarize(s: _Series | None, threshold: float) -> dict:
    """A camera's day in a few numbers, plus its alarm periods for the day lanes."""
    if s is None or not s.ts:
        return {"seconds": 0}
    periods, start, level = [], None, 0
    for t, lv in zip(s.ts, s.level, strict=True):
        if lv != level:
            if level:
                periods.append([start, t, level])
            start, level = t, lv
    if level:
        periods.append([start, s.ts[-1] + 1, level])
    alarm_seconds = sum(1 for lv in s.level if lv)
    return {
        "seconds": len(s.ts),
        "first": s.ts[0],
        "last": s.ts[-1],
        "count_now": round(s.mean[-1], 1),
        "count_mean": round(sum(s.mean) / len(s.mean), 1),
        "count_max": round(max(s.high), 1),
        "above_threshold_seconds": sum(1 for v in s.mean if v >= threshold),
        "alarm_seconds": alarm_seconds,
        "max_level": max(s.level),
        "periods": periods,
        "snapshots": len(s.snapshots),
        "level_now": s.level[-1],
    }


def curve(s: _Series, max_points: int = MAX_POINTS) -> dict:
    """Per-second columns folded into at most `max_points` buckets (mean of means, min, max, max level)."""
    n = len(s.ts)
    step = max(1, -(-n // max_points))
    out = {"step": step, "t": [], "mean": [], "low": [], "high": [], "level": []}
    for i in range(0, n, step):
        j = min(n, i + step)
        out["t"].append(s.ts[i])
        out["mean"].append(round(sum(s.mean[i:j]) / (j - i), 1))
        out["low"].append(round(min(s.low[i:j]), 1))
        out["high"].append(round(max(s.high[i:j]), 1))
        out["level"].append(max(s.level[i:j]))
    return out


# --- labels --------------------------------------------------------------------------------------------


class Labels:
    """labels.csv per day: one row per verdict, the last row of a snapshot wins (an empty label clears)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict[Path, tuple[float, int, dict[str, dict]]] = {}

    def read(self, day_dir: Path) -> dict[str, dict]:
        path = day_dir / countlog.LABELS_FILE
        try:
            st = path.stat()
        except OSError:
            return {}
        with self._lock:
            hit = self._cache.get(path)
            if hit and hit[:2] == (st.st_mtime, st.st_size):
                return hit[2]
            out: dict[str, dict] = {}
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("label"):
                        out[row["snapshot"]] = row
                    else:
                        out.pop(row.get("snapshot", ""), None)
            self._cache[path] = (st.st_mtime, st.st_size, out)
            return out

    def write(self, day_dir: Path, snapshot: str, label: str) -> dict:
        """`snapshot`: a path under the day, or `alarm:<path under the evidence folder>` (checked by the caller)."""
        if label and label not in LABELS:
            raise ValueError(f"label must be one of {', '.join(LABELS)} (or empty to clear)")
        row = {"snapshot": snapshot, "label": label, "labeled_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
        path = day_dir / countlog.LABELS_FILE
        with self._lock:
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
        return row


def label_queue(
    items: list[dict], threshold: float, rated: dict[str, int], labelled: set[str], n: int = QUEUE_PER_CAMERA
) -> dict[str, str]:
    """The few pictures of one camera's day worth a person's look, {key: why}.

    `items`: {key, count, source: "count_log" | "alarm", t}; `rated`: the model's level per key.
    Labelled pictures keep their place, so working through the queue never refills it. The
    other places go first to pictures the count and the model disagree on most (`over`: at the
    threshold or above, rated QUIET_LEVEL or less; `under`: below countlog.HIGH_RATIO of it, rated
    DENSE_LEVEL), then to alarm pictures, then to the snapshots nearest the threshold.
    """

    def why(it: dict) -> tuple[str, float]:  # (reason, rank: lower first)
        level, count = rated.get(it["key"]), it["count"]
        if level is not None and count >= threshold and level <= QUIET_LEVEL:
            return "over", -(count - threshold) / threshold - (QUIET_LEVEL + 1 - level)
        if level is not None and count < countlog.HIGH_RATIO * threshold and level >= DENSE_LEVEL:
            return "under", -(countlog.HIGH_RATIO * threshold - count) / threshold - 1
        if it["source"] == "alarm":
            return "alarm", 10 + it["t"] / 1e10
        return "near", 20 + abs(count - threshold) / threshold

    done = [it for it in items if it["key"] in labelled]
    todo = sorted((it for it in items if it["key"] not in labelled), key=lambda it: (why(it)[1], it["t"]))
    return {it["key"]: why(it)[0] for it in done + todo[: max(0, n - len(done))]}


# --- alarm evidence ------------------------------------------------------------------------------------


class Evidence:
    """events.jsonl of the alarm evidence folder, read incrementally, keyed by camera MAC."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._offset = 0
        self._size = -1
        self._actions: list[dict] = []

    def actions(self) -> list[dict]:
        path = self.root / "events.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            return []
        with self._lock:
            if size < self._size:  # replaced
                self._offset, self._actions = 0, []
            if size != self._size:
                with path.open("rb") as f:
                    f.seek(self._offset)
                    data = f.read()
                end = data.rfind(b"\n") + 1
                for line in data[:end].splitlines():
                    try:
                        a = json.loads(line)
                    except ValueError:
                        continue
                    a["mac"] = mac_of(a.get("camera_id", ""))
                    a["image"] = _relative(self.root, a.get("snapshot_path"))
                    self._actions.append(a)
                self._offset += end
                self._size = size
            return self._actions

    def of_day(self, day: str) -> list[dict]:
        return [a for a in self.actions() if f"{datetime.fromtimestamp(a['timestamp']):%Y-%m-%d}" == day]


def _relative(root: Path, path: str | None) -> str | None:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return rel.as_posix() if rel.suffix.lower() in IMAGE_SUFFIXES else None
