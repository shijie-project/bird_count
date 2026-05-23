import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from runtime.audit import AuditLog


logger = logging.getLogger(__name__)


def writer_loop(
    sid: int,
    q: queue.Queue,
    stop_event: threading.Event,
    fps: float,
    frame_size: tuple,
    segment_seconds: float,
    output_dir: Path,
    audit: Optional[AuditLog],
):
    """Drains `q` into a rotating `cv2.VideoWriter`; finalizes segment on exit."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: Optional[cv2.VideoWriter] = None
    seg_start = 0.0

    def _open(ts: float) -> Optional[cv2.VideoWriter]:
        local = time.localtime(ts)
        date_dir = output_dir / time.strftime("%Y%m%d", local)
        try:
            date_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"[VideoWriter-{sid:02d}] Failed to create dir {date_dir}: {e}")
            return None
        ts_str = time.strftime("%H%M%S", local)
        path = date_dir / f"stream_{sid:02d}_{ts_str}.mp4"
        w = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if not w.isOpened():
            logger.error(f"[VideoWriter-{sid:02d}] Failed to open writer for {path}")
            return None
        logger.info(f"[VideoWriter-{sid:02d}] Recording -> {path}")
        if audit:
            audit.log("recorder.segment_open", stream_id=sid, path=str(path))
        return w

    def _release(w: cv2.VideoWriter):
        try:
            w.release()
        except Exception as e:
            logger.error(f"[VideoWriter-{sid:02d}] release failed: {e}")
            return
        if audit:
            audit.log("recorder.segment_close", stream_id=sid)

    def _write_one(w, seg, frame, ts):
        if w is None or (ts - seg) >= segment_seconds:
            if w is not None:
                _release(w)
            w = _open(ts)
            if w is None:
                return None, 0.0
            seg = ts
        try:
            w.write(frame)
        except Exception as e:
            logger.error(f"[VideoWriter-{sid:02d}] write failed: {e}")
        return w, seg

    try:
        while not stop_event.is_set():
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, ts = item
            writer, seg_start = _write_one(writer, seg_start, frame, ts)

        # Post-stop drain: capture frames already queued so the last ~half-second
        # of footage isn't lost on disable_stream / stop.
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            frame, ts = item
            writer, seg_start = _write_one(writer, seg_start, frame, ts)
    finally:
        if writer is not None:
            _release(writer)
