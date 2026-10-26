"""Ratings on disk, and which pictures still need one.

A rating lands in its day's ratings.csv (countlog.RATINGS_FILE), next to the
snapshots and the person's labels, keyed like a label: the snapshot's path
under the day, or `alarm:<path>` for a picture of the alarm evidence.
Ratings carry the model and the rubric version, so a proposal only ever
mixes ratings made to the same standard.
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from birdcount.deploy import countlog
from birdcount.deploy.alarm.camera_ids import mac_of
from birdcount.deploy.calibrate.rubric import RUBRIC_VERSION


FIELDS = ["snapshot", "mac", "count", "kind", "model", "rubric", "level", "reason", "rated_at"]
# Count-log snapshots rated per camera and 10-bird bin over the look-back window: a week's worth.
PER_BIN = countlog.SAMPLES_PER_BIN * countlog.CALIBRATION_DAYS


@dataclass(frozen=True)
class Candidate:
    day: str
    key: str  # the snapshot's path under the day, or alarm:<path under the evidence folder>
    file: Path
    mac: str
    count: float
    kind: str  # bin / high, or the alarm action (level1, ...)
    t: int


_lock = threading.Lock()


def read(day_dir: Path) -> list[dict]:
    path = day_dir / countlog.RATINGS_FILE
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append(day_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = day_dir / countlog.RATINGS_FILE
    with _lock:
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerows(rows)


def row(c: Candidate, model: str, level: int, reason: str) -> dict:
    return {
        "snapshot": c.key,
        "mac": c.mac,
        "count": f"{c.count:.0f}",
        "kind": c.kind,
        "model": model,
        "rubric": RUBRIC_VERSION,
        "level": level,
        "reason": reason,
        "rated_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    }


def current(rows: list[dict], model: str) -> list[dict]:
    """Ratings made by this model to this rubric; a re-rated picture counts once, the latest."""
    out = {}
    for r in rows:
        if r.get("model") == model and str(r.get("rubric")) == str(RUBRIC_VERSION):
            out[r["snapshot"]] = r
    return list(out.values())


def candidates(root: Path, days: list[str], evidence_root: Path | None) -> list[Candidate]:
    """Every picture of these days that could be rated: count-log snapshots and alarm pictures."""
    out = []
    for day in days:
        day_dir = root / day
        snaps = day_dir / "snapshots"
        if snaps.is_dir():
            for f in sorted(snaps.glob("*/*.jpg")):
                rel = f.relative_to(day_dir).as_posix()
                info = countlog.parse_snapshot(day, rel)
                if info:
                    out.append(Candidate(day, rel, f, info["mac"], info["count"], info["kind"], info["t"]))
    if evidence_root is not None:
        wanted = set(days)
        try:
            lines = (evidence_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                a = json.loads(line)
                path = Path(a["snapshot_path"])
                rel = path.resolve().relative_to(evidence_root.resolve()).as_posix()
            except (ValueError, KeyError, OSError):
                continue
            day = countlog.day_name(a["timestamp"])
            if day in wanted and a.get("action_type", "").startswith("level") and path.is_file():
                key = f"{countlog.ALARM_PREFIX}{rel}"
                out.append(
                    Candidate(
                        day,
                        key,
                        path,
                        mac_of(a["camera_id"]),
                        float(a["count"]),
                        a["action_type"],
                        int(a["timestamp"]),
                    )
                )
    return out


def select(
    cands: list[Candidate], rated: set[str], per_bin: int = PER_BIN, limit: int | None = None
) -> list[Candidate]:
    """What to send now: every unrated alarm picture, and count-log snapshots until each camera's
    10-bird bin holds `per_bin` ratings, spread over the window's days and hours.

    Saturated bins cost nothing more: a camera's low bins fill in a day and are then left alone,
    while the rare high bins keep being rated as pictures of them arrive.
    """
    have: dict[tuple[str, int], int] = {}
    for c in cands:
        if c.key in rated and not c.key.startswith(countlog.ALARM_PREFIX):
            have[(c.mac, countlog.count_bin(c.count))] = have.get((c.mac, countlog.count_bin(c.count)), 0) + 1
    alarms = [c for c in cands if c.key.startswith(countlog.ALARM_PREFIX) and c.key not in rated]
    by_bin: dict[tuple[str, int], list[Candidate]] = {}
    for c in cands:
        if not c.key.startswith(countlog.ALARM_PREFIX) and c.key not in rated:
            by_bin.setdefault((c.mac, countlog.count_bin(c.count)), []).append(c)
    picked = list(alarms)
    for key, group in sorted(by_bin.items()):
        room = per_bin - have.get(key, 0)
        if room <= 0:
            continue
        group.sort(key=lambda c: c.t)
        step = max(1, len(group) // room)
        picked += group[::step][:room]
    return picked[:limit] if limit else picked
