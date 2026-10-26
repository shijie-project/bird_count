"""Off-thread disk I/O for the count logger.

The consumer thread only aggregates and enqueues; this thread owns every file
handle. CSVs stay open per (day, camera) and are flushed every few seconds,
so a 20-camera log costs one write call per camera per second, not an
open/append/close.
"""

import csv
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2


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

QUEUE_MAXSIZE = 4096
_FLUSH_EVERY_SEC = 5.0
_JOIN_TIMEOUT = 5.0
_DROP_LOG_EVERY = 500

# Review snapshots are for a human deciding "pile-up or not"; 640x360 is plenty
# and keeps a 14-hour day of them in the low hundreds of MB.
SNAPSHOT_SIZE = (640, 360)
_JPEG_QUALITY = 80


class CountLogWriter:
    def __init__(self, name: str = "CountLogger"):
        self.name = name
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._thread: Optional[threading.Thread] = None
        self._files: dict[Path, tuple] = {}  # path -> (file, csv.DictWriter)
        self._last_flush = time.monotonic()
        self.drops = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-Writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)  # blocking: the sentinel must land behind every queued row
        self._thread.join(timeout=_JOIN_TIMEOUT)
        if self._thread.is_alive():
            logger.warning("[%s] Writer thread did not finish in %.1fs.", self.name, _JOIN_TIMEOUT)
        self._thread = None
        if self.drops:
            logger.warning("[%s] Dropped %d row/snapshot job(s) on a full queue.", self.name, self.drops)

    # Producer side (consumer thread) -----------------------------------

    def put_row(self, path: Path, row: dict) -> None:
        self._put(("row", path, row))

    def put_snapshot(self, path: Path, frame) -> None:
        self._put(("snap", path, frame))

    def put_json(self, path: Path, text: str) -> None:
        self._put(("json", path, text))

    def _put(self, job) -> None:
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.drops += 1
            if self.drops == 1 or self.drops % _DROP_LOG_EVERY == 0:
                logger.warning("[%s] Write queue full (drops: %d).", self.name, self.drops)

    # Writer thread -----------------------------------------------------

    def _loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._maybe_flush()
                continue
            if job is None:
                break
            try:
                self._do(job)
            except Exception as e:
                logger.error("[%s] Write failed: %s", self.name, e, exc_info=True)
            self._maybe_flush()
        self._close_all()

    def _do(self, job) -> None:
        kind, path, payload = job
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "row":
            entry = self._files.get(path)
            if entry is None:
                new = not path.exists()
                fh = path.open("a", newline="", encoding="utf-8")
                writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
                if new:
                    writer.writeheader()
                entry = self._files[path] = (fh, writer)
                self._close_other_days(path)
            entry[1].writerow(payload)
        elif kind == "snap":
            small = cv2.resize(payload, SNAPSHOT_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), small, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        elif kind == "json":
            path.write_text(payload, encoding="utf-8")

    def _close_other_days(self, current: Path) -> None:
        """A new day's file opened: close the previous days' handles."""
        day_dir = current.parent
        for path in [p for p in self._files if p.parent != day_dir]:
            self._files.pop(path)[0].close()

    def _maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush < _FLUSH_EVERY_SEC:
            return
        for fh, _ in self._files.values():
            fh.flush()
        self._last_flush = time.monotonic()

    def _close_all(self) -> None:
        for fh, _ in self._files.values():
            fh.close()
        self._files.clear()
