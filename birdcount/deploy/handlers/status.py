"""The runtime's live state as one small JSON file, for the farm backend to read.

Rewritten at most once a second, atomically (temp file + rename), so a reader
never sees half of it:

    {"written": <unix time>, "started": ..., "stopped": null | <unix time>,
     "pid": ..., "runtime": {...settings that shape what an alarm does...},
     "cameras": [{"sid", "mac", "label", "zone", "threshold", "count",
                  "captured", "online", "level", "event_id", "alarm_since"}]}

`written` is the heartbeat: a reader that sees it stop moving knows the
runtime is gone, whatever the file says. `stopped` is set by a clean shutdown.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from birdcount.deploy.handlers.base import Batch, Handler
from birdcount.deploy.shm import FrameRing


logger = logging.getLogger(__name__)

WRITE_EVERY_SEC = 1.0


@dataclass
class _Camera:
    sid: int
    mac: str | None
    label: str
    zone: str | None
    threshold: float | None
    count: float | None = None
    captured: float | None = None
    level: int = 0
    event_id: str | None = None
    alarm_since: float | None = None


class StatusFile(Handler):
    name = "Status"

    def __init__(self, path: Path, ring: FrameRing, cameras: list[dict], runtime: dict):
        """`cameras[sid]`: {mac, label, zone, threshold}; `runtime`: what the file reports as is."""
        super().__init__()
        self.path = path
        self.ring = ring
        self.runtime = runtime
        self.cameras = [_Camera(sid=sid, **c) for sid, c in enumerate(cameras)]
        self.started = time.time()
        self._last_write = 0.0
        self._warned = False

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write()

    def stop(self) -> None:
        self.write(stopped=time.time())

    def handle(self, batch: Batch) -> None:
        for i, r in enumerate(batch.records):
            cam = self.cameras[int(r["sid"])]
            cam.count, cam.captured = float(r["count"]), float(r["captured"])
            d = batch.decisions[i]
            level = d.level if d is not None else 0
            if level and not cam.level:
                cam.alarm_since = cam.captured
            elif not level:
                cam.alarm_since = None
            cam.level, cam.event_id = level, d.event_id if d is not None else None

    def tick(self, now: float | None = None) -> None:
        """From the runtime's loop: write when a second has passed, frames or not."""
        now = time.time() if now is None else now
        if now - self._last_write >= WRITE_EVERY_SEC:
            self.write(now=now)

    def write(self, *, now: float | None = None, stopped: float | None = None) -> None:
        now = time.time() if now is None else now
        online = self.ring.status["online"]
        state = {
            "written": now,
            "started": self.started,
            "stopped": stopped,
            "pid": os.getpid(),
            "runtime": self.runtime,
            "cameras": [
                {
                    "sid": c.sid,
                    "mac": c.mac,
                    "label": c.label,
                    "zone": c.zone,
                    "threshold": c.threshold,
                    "count": None if c.count is None else round(c.count, 1),
                    "captured": c.captured,
                    "online": bool(online[c.sid]),
                    "level": c.level,
                    "event_id": c.event_id,
                    "alarm_since": c.alarm_since,
                }
                for c in self.cameras
            ],
        }
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)  # a reader holding the old file open on Windows: retry next second
            self._last_write = now
        except OSError as e:
            if not self._warned:
                logger.warning("Cannot write the status file %s: %s", self.path, e)
                self._warned = True
