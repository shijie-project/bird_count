"""The count log's layout on disk, shared by its writer and its readers.

    <COUNT_LOG_DIR>/<YYYY-MM-DD>/
        meta.json                 model, frame size, hours, thresholds in force
        <MAC>.csv                 one row per camera per second (ROW_FIELDS)
        snapshots/<MAC>/<HHMMSS>_<kind>_<count>.jpg
        labels.csv                a person's verdicts (deploy.webui)
        ratings.csv               a vision model's crowding levels (deploy.calibrate)

The writer is `handlers.count_logger` (in the runtime); the readers are the
farm backend and the calibration, which must not load cv2 or the runtime to
know a file name. stdlib only.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path


ROW_FIELDS = [
    "time_local",
    "timestamp",
    "frames",
    "count_mean",
    "count_max",
    "count_min",
    "centroid_x",
    "centroid_y",
    "alarm_level",
    "motion_excluded_frames",
    "snapshot",
]
DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
SNAPSHOT_RE = re.compile(r"(?P<hms>\d{6})_(?P<kind>[a-z]+)_(?P<count>\d+)\.jpg$")
SAMPLE_STEP = 10  # birds per count bin of the snapshot sampling
HIGH_RATIO = 0.8  # snapshots at this share of the threshold or more are `high`: the counts that decide it
# Snapshots start at a camera's floor: FLOOR_RATIO of its threshold, rounded up to a bin.
# Below it the shed is at ease, and its pictures tell nothing about where the threshold goes.
FLOOR_RATIO = 0.6
SAMPLES_PER_BIN = 3  # snapshots per camera, bin and day from the floor up
CALIBRATION_DAYS = 7  # a week of them is what a threshold suggestion is made from
LABELS_FILE = "labels.csv"
RATINGS_FILE = "ratings.csv"
ALARM_PREFIX = "alarm:"  # how labels and ratings name a picture of the alarm evidence


def floor_of(threshold: float | None) -> int:
    """The lowest count a camera keeps snapshots of; every count when it has no threshold."""
    if threshold is None:
        return 0
    return math.ceil(FLOOR_RATIO * threshold / SAMPLE_STEP) * SAMPLE_STEP


def day_name(ts: float) -> str:
    return f"{datetime.fromtimestamp(ts):%Y-%m-%d}"


def snapshot_path(mac: str, ts: float, kind: str, count: float) -> str:
    """A snapshot's path relative to its day."""
    return f"snapshots/{mac}/{datetime.fromtimestamp(ts):%H%M%S}_{kind}_{count:.0f}.jpg"


def parse_snapshot(day: str, rel: str) -> dict | None:
    """What a snapshot's name says: when, which kind of sample, the count it was taken at, the camera."""
    m = SNAPSHOT_RE.search(rel)
    if not m:
        return None
    t = datetime.strptime(f"{day} {m['hms']}", "%Y-%m-%d %H%M%S").timestamp()
    parts = Path(rel).parts
    mac = parts[-2] if len(parts) >= 2 else None
    return {"path": rel, "t": int(t), "kind": m["kind"], "count": int(m["count"]), "mac": mac}


def count_bin(count: float) -> int:
    return int(max(count, 0.0) // SAMPLE_STEP)


def days(root: Path | None) -> list[str]:
    """The days with a count log, newest first."""
    if root is None or not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if p.is_dir() and DAY_RE.fullmatch(p.name)), reverse=True)


def file_sha256(path: Path) -> str:
    """A model file's sha256, which meta.json records: counts are only comparable under one model. "" if unreadable."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""
