import logging
import queue
import threading
import time
from pathlib import Path

import cv2

from runtime.audit import AuditLog


logger = logging.getLogger(__name__)

# If the primary fourcc (e.g. "avc1") can't open on this machine — usually
# because the H.264 backend isn't installed — fall back to mp4v so recording
# still works, just with worse compression.
_FALLBACK_FOURCC = "mp4v"


def writer_loop(
    sid: int,
    q: queue.Queue,
    stop_event: threading.Event,
    fps: float,
    frame_size: tuple[int, int],
    segment_seconds: float,
    output_dir: Path,
    fourcc_name: str,
    audit: AuditLog,
):
    """Drains `q` into a rotating `cv2.VideoWriter`; finalizes segment on exit.

    `audit` is either a real `AuditLog` or `BaseHandler`'s `_NullAudit` stub —
    both duck-type the same `.log(...)` call, so no None-guarding here.
    """
    primary = cv2.VideoWriter_fourcc(*fourcc_name)
    fallback = cv2.VideoWriter_fourcc(*_FALLBACK_FOURCC)
    active_fourcc = primary
    active_name = fourcc_name
    writer: cv2.VideoWriter | None = None
    seg_start = 0.0

    stream_root = output_dir / f"stream_{sid:02d}"

    def _open(ts: float) -> cv2.VideoWriter | None:
        nonlocal active_fourcc, active_name
        local = time.localtime(ts)
        date_dir = stream_root / time.strftime("%Y%m%d", local)
        try:
            date_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("[VideoWriter-%02d] Failed to create dir %s: %s", sid, date_dir, e, exc_info=True)
            return None
        ts_str = time.strftime("%Y%m%d-%H%M%S", local)
        ext = ".avi" if active_name.lower() == "mjpg" else ".mp4"
        path = date_dir / f"{ts_str}{ext}"
        w = cv2.VideoWriter(str(path), active_fourcc, fps, frame_size)
        if not w.isOpened() and active_fourcc != fallback:
            logger.warning(
                "[VideoWriter-%02d] fourcc '%s' not available; falling back to '%s' for the rest of this session.",
                sid,
                active_name,
                _FALLBACK_FOURCC,
            )
            active_fourcc = fallback
            active_name = _FALLBACK_FOURCC
            # _FALLBACK_FOURCC is "mp4v"; the .mp4 extension below matches.
            path = date_dir / f"{ts_str}.mp4"
            w = cv2.VideoWriter(str(path), active_fourcc, fps, frame_size)
        if not w.isOpened():
            logger.error("[VideoWriter-%02d] Failed to open writer for %s", sid, path)
            return None
        logger.info("[VideoWriter-%02d] Recording -> %s  (%s)", sid, path, active_name)
        audit.log("recorder.segment_open", stream_id=sid, path=str(path), fourcc=active_name)
        return w

    def _release(w: cv2.VideoWriter):
        try:
            w.release()
        except Exception as e:
            logger.error("[VideoWriter-%02d] release failed: %s", sid, e, exc_info=True)
            return
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
            logger.error("[VideoWriter-%02d] write failed: %s", sid, e, exc_info=True)
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
