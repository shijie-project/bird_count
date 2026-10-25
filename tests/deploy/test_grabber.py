import time

import numpy as np
import pytest

from birdcount.deploy.grabber import StreamReader
from birdcount.deploy.shm import FrameRing
from birdcount.deploy.sources import Source

from .conftest import make_layout, write_video


def _run(reader: StreamReader, seconds: float) -> None:
    reader.start()
    time.sleep(seconds)
    reader.stop()
    reader.join(timeout=5)
    assert not reader.is_alive()


def test_an_open_monitor_gets_frames_at_its_rate(tmp_path) -> None:
    ring = FrameRing.allocate(make_layout(streams=1), fps=1)
    try:
        src = Source(0, str(write_video(tmp_path / "a.avi", 20)), is_file=True)  # 25 fps
        ring.display_fps = 100  # faster than the clip plays: every frame is wanted
        reader = StreamReader(src, ring, hw_decode=False, realtime=True)
        _run(reader, 1.5)
        assert reader.decoded > 20  # looped past the end
        assert reader.frame_idx >= reader.decoded  # every frame (+ the NO SIGNAL placeholder)

        ring.display_fps = 5  # slower than the clip: only display ticks
        reader = StreamReader(src, ring, hw_decode=False, realtime=True)
        _run(reader, 1.5)
        assert reader.decoded >= 30 and reader.frame_idx <= 1 + 1.5 * 5 + 2
        slot, _ = ring.newest_ready(0)
        assert ring.frames[0, slot].std() < 2  # a real, uniform video frame
    finally:
        ring.close()


def test_without_display_only_tick_frames_are_retrieved(tmp_path) -> None:
    ring = FrameRing.allocate(make_layout(streams=1), fps=4)  # ticks every 0.25 s
    try:
        src = Source(0, str(write_video(tmp_path / "a.avi", 50)), is_file=True)
        reader = StreamReader(src, ring, hw_decode=False, realtime=True)  # 25 fps playback
        _run(reader, 1.6)
        assert reader.decoded >= 30
        assert reader.frame_idx <= 1 + 1.6 * 4 + 2  # NO SIGNAL + about one per tick
    finally:
        ring.close()


def test_frames_are_fitted_to_the_ring_geometry(tmp_path) -> None:
    ring = FrameRing.allocate(make_layout(streams=1), fps=50)
    try:
        src = Source(0, str(write_video(tmp_path / "big.avi", 10, size=(192, 128))), is_file=True)
        _run(StreamReader(src, ring, hw_decode=False, realtime=False), 0.5)
        slot, idx = ring.newest_ready(0)
        assert idx > 1 and ring.frames[0, slot].std() < 2
    finally:
        ring.close()


def test_a_dead_source_shows_no_signal_and_retries(tmp_path) -> None:
    ring = FrameRing.allocate(make_layout(streams=1), fps=10)
    try:
        src = Source(0, str(tmp_path / "missing.avi"), is_file=True)
        reader = StreamReader(src, ring, hw_decode=False, backoff_base=0.05)
        _run(reader, 0.6)
        assert reader.frame_idx >= 2  # one NO SIGNAL frame per attempt
        slot, _ = ring.newest_ready(0)
        assert ring.frames[0, slot].max() > 100 and ring.frames[0, slot].mean() < 50  # text on black
    finally:
        ring.close()


@pytest.mark.parametrize("hw", [True, False])
def test_hardware_preference_never_loses_a_stream(tmp_path, hw) -> None:
    from birdcount.deploy.sources import open_capture

    cap = open_capture(Source(0, str(write_video(tmp_path / "a.avi", 5)), is_file=True), hw_decode=hw)
    assert cap is not None and cap.fps == pytest.approx(25.0)
    ok, frame = cap.cap.read()
    cap.cap.release()
    assert ok and np.asarray(frame).shape == (64, 96, 3)
