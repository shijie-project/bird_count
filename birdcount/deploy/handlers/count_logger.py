"""Daily per-camera count log, for re-tuning the alarm thresholds on farm data.

Inside the configured local-time window every camera gets one CSV row per
second, and snapshots are kept across the whole count range, so each
camera's count can later be matched to how crowded its picture looks (by a
person or a vision model):

    <COUNT_LOG_DIR>/<YYYY-MM-DD>/
        meta.json               model + sha256, frame size, thresholds in force
        <MAC>.csv               one row per second (ROW_FIELDS)
        snapshots/<MAC>/<HHMMSS>_<kind>_<count>.jpg

Snapshots are sampled per count bin of SAMPLE_STEP birds: up to
SAMPLES_PER_BIN a day per camera and bin, SAMPLE_GAP_SEC apart, so a day
holds pictures of every level the camera reached instead of hundreds of the
commonest one. Bins from 80 % of the threshold up are kept more densely
(`high`: the counts that decide the threshold); the rest are `bin`. A row
names the snapshot taken in its second, so CSV and pictures join on that
column. Rows carry the engine's alarm level and motion exclusions, so
replaying a day through the engine with candidate thresholds is a plain
offline loop.

The consumer thread aggregates and shrinks snapshots straight out of the frame
ring (no full-frame copy); a writer thread owns every file handle and keeps
the day's CSVs open, flushing every few seconds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.base import Batch, Handler
from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.settings import parse_hours


logger = logging.getLogger(__name__)

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
SAMPLE_STEP = 10  # birds per count bin
SAMPLES_PER_BIN = 6  # per camera, bin and day
SAMPLE_GAP_SEC = 10 * 60.0  # between two pictures of one bin, so they show different moments
HIGH_SNAPSHOT_RATIO = 0.8  # bins from this fraction of the threshold up are `high`
HIGH_SAMPLES_PER_BIN = 30
HIGH_SAMPLE_GAP_SEC = 60.0
# Review snapshots are for a person deciding "pile-up or not": 640x360 is
# plenty, and keeps a 14-hour day in the low hundreds of MB.
SNAPSHOT_SIZE = (640, 360)
JPEG_QUALITY = 80
QUEUE_MAXSIZE = 4096
_FLUSH_EVERY_SEC = 5.0


@dataclass
class _Second:
    """One camera's samples within one wall-clock second."""

    second: int
    counts: list[float] = field(default_factory=list)
    cx: list[float] = field(default_factory=list)
    cy: list[float] = field(default_factory=list)
    level: int = 0
    excluded: int = 0
    snapshot: str = ""


class _Writer(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="CountLogWriter", daemon=True)
        self.queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._files: dict[Path, tuple] = {}
        self._last_flush = time.monotonic()
        self.drops = 0

    def put(self, kind: str, path: Path, payload) -> None:
        try:
            self.queue.put_nowait((kind, path, payload))
        except queue.Full:
            self.drops += 1
            if self.drops == 1 or self.drops % 500 == 0:
                logger.warning("Count log write queue full (drops: %d).", self.drops)

    def run(self) -> None:
        while True:
            try:
                job = self.queue.get(timeout=1.0)
            except queue.Empty:
                self._flush()
                continue
            if job is None:
                break
            try:
                self._write(*job)
            except Exception:
                logger.exception("Count log write failed.")
            self._flush()
        for fh, _ in self._files.values():
            fh.close()

    def _write(self, kind: str, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "row":
            if path not in self._files:
                for old in [p for p in self._files if p.parent != path.parent]:  # a new day: close the old ones
                    self._files.pop(old)[0].close()
                new = not path.exists()
                fh = path.open("a", newline="", encoding="utf-8")
                writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
                if new:
                    writer.writeheader()
                self._files[path] = (fh, writer)
            self._files[path][1].writerow(payload)
        elif kind == "jpg":
            cv2.imwrite(str(path), payload, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        else:
            path.write_text(payload, encoding="utf-8")

    def _flush(self) -> None:
        if time.monotonic() - self._last_flush >= _FLUSH_EVERY_SEC:
            for fh, _ in self._files.values():
                fh.flush()
            self._last_flush = time.monotonic()


class CountLogger(Handler):
    name = "CountLogger"

    def __init__(
        self,
        root: Path,
        hours: str,
        engine: AlarmEngine,
        macs: Sequence[str | None],
        *,
        meta: dict,
        model_path: Path | None,
        audit: AuditLog = NULL_AUDIT,
    ):
        """`meta`: run settings recorded in each day's meta.json (frame size, fps, masks)."""
        super().__init__(audit)
        self.root = root
        self.start_t, self.end_t = parse_hours(hours)
        self.labels = {sid: mac or f"sid{sid}" for sid, mac in enumerate(macs)}
        self.thresholds = {sid: t for sid in self.labels if (t := engine.threshold_of(sid)) is not None}
        self._meta = meta
        self._model_path = model_path
        self._model_sha = None
        self._seconds: dict[int, _Second] = {}
        self._bins: dict[tuple[int, int], tuple[int, float]] = {}  # (sid, bin) -> (taken today, last time)
        self._bins_day = ""
        self._meta_days: set[str] = set()
        self._writer: _Writer | None = None

    def start(self) -> None:
        self._writer = _Writer()
        self._writer.start()
        logger.info(
            "Count log: %d camera(s), %s-%s local time, in %s.",
            len(self.labels),
            f"{self.start_t:%H:%M}",
            f"{self.end_t:%H:%M}",
            self.root,
        )

    def stop(self) -> None:
        if self._writer is None:
            return
        for sid in list(self._seconds):
            self._flush(sid)
        self._writer.queue.put(None)  # blocking: lands behind every queued job
        self._writer.join(timeout=5.0)
        if self._writer.drops:
            logger.warning("Count log dropped %d job(s) on a full queue.", self._writer.drops)
        self._writer = None

    def in_window(self, ts: float) -> bool:
        return self.start_t <= datetime.fromtimestamp(ts).time() < self.end_t

    def handle(self, batch: Batch) -> None:
        if self._writer is None:
            return
        for i, r in enumerate(batch.records):
            sid, ts, count = int(r["sid"]), float(r["captured"]), float(r["count"])
            sec = self._seconds.get(sid)
            if sec is not None and sec.second != int(ts):
                self._flush(sid)
                sec = None
            if not self.in_window(ts):
                continue
            if sec is None:
                sec = self._seconds[sid] = _Second(second=int(ts))
            sec.counts.append(count)
            if r["cx"] == r["cx"]:  # not NaN: the frame has a centroid
                sec.cx.append(float(r["cx"]))
                sec.cy.append(float(r["cy"]))
            d = batch.decisions[i]
            if d is not None:
                sec.level = max(sec.level, d.level)
                sec.excluded += int(d.sample.motion_excluded)
            kind = self._snapshot_kind(sid, ts, count)
            if kind and not sec.snapshot:
                rel = f"snapshots/{self.labels[sid]}/{datetime.fromtimestamp(ts):%H%M%S}_{kind}_{count:.0f}.jpg"
                small = cv2.resize(batch.frame(i), SNAPSHOT_SIZE, interpolation=cv2.INTER_AREA)
                self._writer.put("jpg", self._day_dir(ts) / rel, small)
                sec.snapshot = rel

    def _snapshot_kind(self, sid: int, ts: float, count: float) -> str:
        day = f"{datetime.fromtimestamp(ts):%Y-%m-%d}"
        if day != self._bins_day:
            self._bins, self._bins_day = {}, day
        threshold = self.thresholds.get(sid)
        high = threshold is not None and count >= HIGH_SNAPSHOT_RATIO * threshold
        cap, gap = (HIGH_SAMPLES_PER_BIN, HIGH_SAMPLE_GAP_SEC) if high else (SAMPLES_PER_BIN, SAMPLE_GAP_SEC)
        key = (sid, int(max(count, 0.0) // SAMPLE_STEP))
        taken, last = self._bins.get(key, (0, -1e18))
        if taken >= cap or ts - last < gap:
            return ""
        self._bins[key] = (taken + 1, ts)
        return "high" if high else "bin"

    def _flush(self, sid: int) -> None:
        sec = self._seconds.pop(sid, None)
        if sec is None or not sec.counts:
            return
        c = sec.counts
        row = {
            "time_local": f"{datetime.fromtimestamp(sec.second):%Y-%m-%d %H:%M:%S}",
            "timestamp": sec.second,
            "frames": len(c),
            "count_mean": f"{sum(c) / len(c):.2f}",
            "count_max": f"{max(c):.2f}",
            "count_min": f"{min(c):.2f}",
            "centroid_x": f"{sum(sec.cx) / len(sec.cx):.4f}" if sec.cx else "",
            "centroid_y": f"{sum(sec.cy) / len(sec.cy):.4f}" if sec.cy else "",
            "alarm_level": sec.level,
            "motion_excluded_frames": sec.excluded,
            "snapshot": sec.snapshot,
        }
        day = self._day_dir(sec.second)
        self._ensure_meta(day)
        self._writer.put("row", day / f"{self.labels[sid]}.csv", row)

    def _day_dir(self, ts: float) -> Path:
        return self.root / f"{datetime.fromtimestamp(ts):%Y-%m-%d}"

    def _ensure_meta(self, day: Path) -> None:
        """What produced the day's counts: thresholds only mean anything per model."""
        if day.name in self._meta_days:
            return
        self._meta_days.add(day.name)
        if self._model_sha is None:
            self._model_sha = _sha256(self._model_path) if self._model_path else ""
        meta = {
            "model_path": str(self._model_path or ""),
            "model_sha256": self._model_sha,
            **self._meta,
            "hours": f"{self.start_t:%H:%M}-{self.end_t:%H:%M}",
            "thresholds": {self.labels[sid]: t for sid, t in sorted(self.thresholds.items())},
            "streams": {str(sid): label for sid, label in self.labels.items()},
        }
        self._writer.put("json", day / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))


def _sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""
