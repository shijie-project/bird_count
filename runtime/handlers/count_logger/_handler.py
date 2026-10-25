"""CountLoggerHandler — daily per-camera count log for re-tuning thresholds.

Inside the configured hours (local time, `COUNT_LOG_HOURS`, 06:00-20:00 by
default) every camera gets one CSV row per second, and review snapshots are
saved so a human can later mark moments as pile-up or not:

    <COUNT_LOG_DIR>/<YYYY-MM-DD>/
        meta.json               model file + sha256, frame size, thresholds in force
        <MAC>.csv               one row per second (see _writer.ROW_FIELDS)
        snapshots/<MAC>/<HHMMSS>_<kind>_<count>.jpg

Snapshot kinds: `high` whenever the count reaches 80 % of the camera's alarm
threshold (at most one a minute — the candidates worth labelling), `routine`
every 15 minutes (the "normal" examples). A row names the snapshot taken in
its second, so the CSV and the pictures join on that column.

The rows are built from the alarm engine's annotations, so `alarm_level` and
`motion_excluded_frames` show what the alarm made of that second; replaying
the logged counts through `_AlarmEngine` with candidate thresholds is then a
plain offline loop.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Optional

import numpy as np

from alarm import load_config
from alarm.camera_ids import resolve_camera_id
from runtime.config import Config
from runtime.handlers import BaseHandler
from runtime.inferencer import BatchInferenceResult, InferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._writer import CountLogWriter


logger = logging.getLogger(__name__)


HIGH_SNAPSHOT_RATIO = 0.8
HIGH_SNAPSHOT_GAP_SEC = 60.0
ROUTINE_SNAPSHOT_GAP_SEC = 15 * 60.0


def parse_hours(spec: str) -> tuple[dtime, dtime]:
    """'06:00-20:00' -> (06:00, 20:00). The end is exclusive."""
    try:
        start, end = (datetime.strptime(part.strip(), "%H:%M").time() for part in spec.split("-"))
    except ValueError as e:
        raise ValueError(f"COUNT_LOG_HOURS must look like 'HH:MM-HH:MM', got {spec!r}") from e
    if start >= end:
        raise ValueError(f"COUNT_LOG_HOURS start must be before end, got {spec!r}")
    return start, end


@dataclass
class _Bucket:
    """One camera's samples within one wall-clock second."""

    second: int
    counts: list = field(default_factory=list)
    cx: list = field(default_factory=list)
    cy: list = field(default_factory=list)
    level: int = 0
    excluded: int = 0
    snapshot: str = ""


class CountLoggerHandler(BaseHandler):
    needs_frames = False  # a frame is copied by hand, only when a snapshot is due

    def __init__(self, config: Config, shm_config: SharedMemoryConfig, name: str = "CountLogger"):
        super().__init__(config=config, shm_config=shm_config, name=name)
        self._enabled = bool(config.envs.enable_count_logger)
        self.root = Path(config.envs.count_log_dir)
        self.start_t, self.end_t = parse_hours(config.envs.count_log_hours)

        self._labels = {sid: (config.sid_to_mac.get(sid) or f"sid{sid}") for sid in range(config.num_streams)}
        self._thresholds = self._load_thresholds(config)
        self._buckets: dict[int, _Bucket] = {}
        self._last_high: dict[int, float] = {}
        self._last_routine: dict[int, float] = {}
        self._meta_days: set[str] = set()
        self._writer: Optional[CountLogWriter] = None

    def _load_thresholds(self, config: Config) -> dict[int, float]:
        try:
            cameras = load_config(config.envs.alarm_config_path).cameras
        except Exception as e:
            logger.warning("[%s] No alarm thresholds (%s); only routine snapshots will be taken.", self.name, e)
            return {}
        out = {}
        for sid in range(config.num_streams):
            camera_id = resolve_camera_id(mac=config.sid_to_mac.get(sid), known=cameras.keys())
            if camera_id is not None:
                out[sid] = cameras[camera_id].threshold
        return out

    # Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        super().start()
        if not self._enabled:
            return
        self._writer = CountLogWriter(self.name)
        self._writer.start()
        logger.info(
            "[%s] Logging %d stream(s) %s-%s local time to %s.",
            self.name,
            len(self._labels),
            self.start_t.strftime("%H:%M"),
            self.end_t.strftime("%H:%M"),
            self.root,
        )

    def stop(self) -> None:
        if self._writer is not None:
            for sid in list(self._buckets):
                self._flush(sid)
            self._writer.stop()
            self._writer = None
        super().stop()

    # Processing --------------------------------------------------------

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        pass  # handle_batch is overridden

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemory) -> set[tuple[int, int]]:
        if not self._enabled or self._writer is None:
            return set()
        for result in batch_result.results:
            try:
                self._observe(result, shm_client)
            except Exception as e:
                logger.error("[%s] Logging failed for sid=%s: %s", self.name, result.stream_id, e, exc_info=True)
        return set()

    def in_window(self, ts: float) -> bool:
        return self.start_t <= datetime.fromtimestamp(ts).time() < self.end_t

    def _observe(self, result: InferenceResult, shm_client: SharedMemory) -> None:
        sid, ts = int(result.stream_id), float(result.timestamp)
        bucket = self._buckets.get(sid)
        if bucket is not None and bucket.second != int(ts):
            self._flush(sid)
            bucket = None
        if not self.in_window(ts):
            return
        if bucket is None:
            bucket = self._buckets[sid] = _Bucket(second=int(ts))

        bucket.counts.append(float(result.count))
        if result.centroid_x is not None:
            bucket.cx.append(result.centroid_x)
            bucket.cy.append(result.centroid_y)
        bucket.level = max(bucket.level, int(result.alarm_level))
        sample = result.alarm_sample
        if sample is not None and sample.motion_excluded:
            bucket.excluded += 1

        kind = self._snapshot_kind(sid, ts, float(result.count))
        if kind and not bucket.snapshot:
            frame = shm_client.frames[sid, int(result.buffer_idx)].copy()
            local = datetime.fromtimestamp(ts)
            rel = f"snapshots/{self._labels[sid]}/{local:%H%M%S}_{kind}_{result.count:.0f}.jpg"
            self._writer.put_snapshot(self._day_dir(local) / rel, frame)
            bucket.snapshot = rel

    def _snapshot_kind(self, sid: int, ts: float, count: float) -> str:
        threshold = self._thresholds.get(sid)
        if threshold and count >= HIGH_SNAPSHOT_RATIO * threshold:
            if ts - self._last_high.get(sid, -1e18) >= HIGH_SNAPSHOT_GAP_SEC:
                self._last_high[sid] = ts
                return "high"
        if ts - self._last_routine.get(sid, -1e18) >= ROUTINE_SNAPSHOT_GAP_SEC:
            self._last_routine[sid] = ts
            return "routine"
        return ""

    def _flush(self, sid: int) -> None:
        bucket = self._buckets.pop(sid, None)
        if bucket is None or not bucket.counts:
            return
        local = datetime.fromtimestamp(bucket.second)
        counts = bucket.counts
        row = {
            "time_local": f"{local:%Y-%m-%d %H:%M:%S}",
            "timestamp": bucket.second,
            "frames": len(counts),
            "count_mean": f"{sum(counts) / len(counts):.2f}",
            "count_max": f"{max(counts):.2f}",
            "count_min": f"{min(counts):.2f}",
            "centroid_x": f"{sum(bucket.cx) / len(bucket.cx):.4f}" if bucket.cx else "",
            "centroid_y": f"{sum(bucket.cy) / len(bucket.cy):.4f}" if bucket.cy else "",
            "alarm_level": bucket.level,
            "motion_excluded_frames": bucket.excluded,
            "snapshot": bucket.snapshot,
        }
        day = self._day_dir(local)
        self._ensure_meta(day)
        self._writer.put_row(day / f"{self._labels[sid]}.csv", row)

    def _day_dir(self, local: datetime) -> Path:
        return self.root / f"{local:%Y-%m-%d}"

    def _ensure_meta(self, day: Path) -> None:
        """Record what produced the day's counts: thresholds only mean anything per model."""
        if day.name in self._meta_days:
            return
        self._meta_days.add(day.name)
        model = Path(self.config.envs.model_path)
        meta = {
            "model_path": str(model),
            "model_sha256": _sha256(model),
            "frame_size": [self.config.envs.image_width, self.config.envs.image_height],
            "mask_dir": self.config.envs.mask_dir,
            "fps": self.config.envs.fps,
            "hours": f"{self.start_t:%H:%M}-{self.end_t:%H:%M}",
            "thresholds": {self._labels[sid]: t for sid, t in sorted(self._thresholds.items())},
            "streams": {str(sid): label for sid, label in self._labels.items()},
        }
        self._writer.put_json(day / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))


def _sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""
