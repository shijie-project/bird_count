"""Continuous recording: every enabled camera into fixed-length video segments.

One encoder thread per recording camera. `handle` copies each enabled
camera's frame out of the ring (its slot is released right after) onto that
camera's queue; the thread writes it into the current segment and starts a
new file every `segment_seconds`:

    <VIDEO_RECORD_DIR>/stream_<sid>/<YYYYMMDD>/<YYYYMMDD-HHMMSS>.mp4

Encoding and the copy release the GIL, so cameras encode in parallel, bounded
by disk. A camera whose encoder falls behind drops frames (logged) rather than
growing memory. A fourcc this machine cannot open falls back to mp4v.
Segments are recorded at the inference rate: the frames the counts came from.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.base import Batch, Handler


logger = logging.getLogger(__name__)

QUEUE_MAXSIZE = 30  # ~83 MB of 720p frames per camera
JOIN_TIMEOUT = 5.0
DROP_LOG_EVERY = 100
_FALLBACK_FOURCC = "mp4v"


class _SegmentWriter(threading.Thread):
    """Drains one camera's queue into rotating segments; finishes the last one on stop."""

    def __init__(
        self,
        sid: int,
        root: Path,
        fps: float,
        size: tuple[int, int],
        segment_seconds: float,
        fourcc: str,
        audit: AuditLog,
    ):
        super().__init__(name=f"Recorder-{sid:02d}", daemon=True)
        self.sid = sid
        self.root = root / f"stream_{sid:02d}"
        self.fps, self.size, self.segment_seconds = fps, size, segment_seconds
        self.fourcc = fourcc
        self.audit = audit
        self.queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.drops = 0
        self._writer: cv2.VideoWriter | None = None
        self._segment_start = 0.0

    def put(self, frame: np.ndarray, ts: float) -> None:
        try:
            self.queue.put_nowait((frame, ts))
        except queue.Full:
            self.drops += 1
            if self.drops == 1 or self.drops % DROP_LOG_EVERY == 0:
                logger.warning("Recording camera %d: encoder behind, %d frame(s) dropped.", self.sid, self.drops)

    def finish(self) -> None:
        self.queue.put(None)  # blocking: everything queued before it is still written
        self.join(timeout=JOIN_TIMEOUT)
        if self.is_alive():
            logger.warning(
                "Recording camera %d did not finish in %.0f s; the last segment may be cut.", self.sid, JOIN_TIMEOUT
            )

    def run(self) -> None:
        try:
            while (item := self.queue.get()) is not None:
                frame, ts = item
                if self._writer is None or ts - self._segment_start >= self.segment_seconds:
                    self._rotate(ts)
                if self._writer is not None:
                    self._writer.write(frame)
        finally:
            self._close()

    def _rotate(self, ts: float) -> None:
        self._close()
        local = time.localtime(ts)
        folder = self.root / time.strftime("%Y%m%d", local)
        folder.mkdir(parents=True, exist_ok=True)
        for name in dict.fromkeys((self.fourcc, _FALLBACK_FOURCC)):
            path = folder / f"{time.strftime('%Y%m%d-%H%M%S', local)}{'.avi' if name.lower() == 'mjpg' else '.mp4'}"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*name), self.fps, self.size)
            if writer.isOpened():
                if name != self.fourcc:
                    logger.warning("Codec %s unavailable; recording with %s.", self.fourcc, name)
                    self.fourcc = name
                self._writer, self._segment_start = writer, ts
                self.audit.log("recorder.segment_open", stream_id=self.sid, path=str(path), fourcc=name)
                return
        logger.error("Cannot open a video writer for camera %d in %s.", self.sid, folder)

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.audit.log("recorder.segment_close", stream_id=self.sid)


class RecorderHandler(Handler):
    name = "Recorder"

    def __init__(
        self,
        num_streams: int,
        root: Path,
        *,
        fps: float,
        frame_size: tuple[int, int],
        segment_seconds: float,
        fourcc: str,
        enabled: bool,
        audit: AuditLog = NULL_AUDIT,
    ):
        super().__init__(audit)
        self.num_streams = num_streams
        self.root = root
        self._args = (fps, frame_size, segment_seconds, fourcc)
        self._start_enabled = enabled
        self._writers: dict[int, _SegmentWriter] = {}

    def start(self) -> None:
        if self._start_enabled:
            self.set_all(True)

    def stop(self) -> None:
        self.set_all(False)

    def handle(self, batch: Batch) -> None:
        if not self._writers:
            return
        now = time.time()
        for i, sid in enumerate(batch.records["sid"].tolist()):
            writer = self._writers.get(sid)
            if writer is not None:
                writer.put(batch.frame(i).copy(), now)

    # Toggles --------------------------------------------------------------------

    def is_recording(self, sid: int) -> bool:
        return sid in self._writers

    def set_recording(self, sid: int, on: bool) -> None:
        if on and sid not in self._writers:
            self._writers[sid] = _SegmentWriter(sid, self.root, *self._args, self.audit)
            self._writers[sid].start()
            self.audit.log("recorder.stream_enable", stream_id=sid)
        elif not on and sid in self._writers:
            self._writers.pop(sid).finish()
            self.audit.log("recorder.stream_disable", stream_id=sid)

    def any_recording(self) -> bool:
        return bool(self._writers)

    def set_all(self, on: bool) -> None:
        for sid in range(self.num_streams):
            self.set_recording(sid, on)
