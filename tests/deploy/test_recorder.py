"""Continuous recording: per-camera switches, segment rotation, a clean finish."""

import time
from types import SimpleNamespace

import cv2
import numpy as np

from birdcount.deploy.audit import NULL_AUDIT
from birdcount.deploy.handlers.base import Batch
from birdcount.deploy.recorder import RecorderHandler, _SegmentWriter
from birdcount.deploy.results import empty_batch


SIZE = (128, 72)


def frame_count(path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        n = 0
        while cap.read()[0]:
            n += 1
        return n
    finally:
        cap.release()


def recorder(root, fourcc: str = "mp4v", **kw) -> RecorderHandler:
    return RecorderHandler(2, root, fps=10, frame_size=SIZE, segment_seconds=300, fourcc=fourcc, enabled=False, **kw)


def batch(ring, sids) -> Batch:
    r = empty_batch(len(sids))
    r["sid"], r["slot"] = sids, 0
    return Batch(r, [None] * len(sids), frozenset(), ring)


def test_only_switched_on_cameras_are_recorded_and_every_frame_lands(tmp_path) -> None:
    ring = SimpleNamespace(frames=np.full((2, 1, SIZE[1], SIZE[0], 3), 90, np.uint8))
    rec = recorder(tmp_path)
    rec.set_recording(1, True)
    assert rec.is_recording(1) and not rec.is_recording(0) and rec.any_recording()
    for _ in range(12):
        rec.handle(batch(ring, [0, 1]))
    rec.stop()  # drains the queue and closes the segment
    assert not rec.any_recording()
    assert not (tmp_path / "stream_00").exists()
    (segment,) = (tmp_path / "stream_01").rglob("*.mp4")
    assert segment.parent.name == time.strftime("%Y%m%d")
    assert frame_count(segment) == 12


def test_segments_rotate_on_time(tmp_path) -> None:
    writer = _SegmentWriter(0, tmp_path, 10, SIZE, 2.0, "mp4v", NULL_AUDIT)
    writer.start()
    frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    t0 = time.time()
    for offset in (0.0, 0.5, 1.5, 3.1, 3.5):  # a new segment at 3.1 s
        writer.put(frame, t0 + offset)
    writer.finish()
    segments = sorted((tmp_path / "stream_00").rglob("*.mp4"))
    assert [frame_count(p) for p in segments] == [3, 2]


def test_an_unknown_codec_falls_back_to_mp4v(tmp_path) -> None:
    ring = SimpleNamespace(frames=np.zeros((2, 1, SIZE[1], SIZE[0], 3), np.uint8))
    rec = recorder(tmp_path, fourcc="ZZZZ")
    rec.set_all(True)
    rec.handle(batch(ring, [0]))
    rec.set_all(False)
    (segment,) = (tmp_path / "stream_00").rglob("*.mp4")
    assert frame_count(segment) == 1
