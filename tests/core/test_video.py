from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from birdcount.core import video as reader
from birdcount.core.video import (
    SampledFrames,
    VideoInfo,
    find_frame,
    fingerprint,
    probe,
    sampling_step,
    start_from_name,
)


SIZE = (96, 64)  # (width, height)
FPS = 10.0


@pytest.fixture(scope="module")
def clip(tmp_path_factory, make_video) -> tuple[Path, list[np.ndarray]]:
    path = tmp_path_factory.mktemp("video") / "B8A44FD52B7F_20260312_103023_ADA3.mp4"
    return path, make_video(path, 60, SIZE, FPS)


def test_start_time_comes_from_the_recording_name() -> None:
    assert start_from_name("B8A44FD52B7F_20260312_103023_ADA3.mkv") == datetime(2026, 3, 12, 10, 30, 23)
    assert start_from_name("20260311_091325_972C.mkv") == datetime(2026, 3, 11, 9, 13, 25)  # AXIS recorder
    assert start_from_name("20260801-061504.mp4") == datetime(2026, 8, 1, 6, 15, 4)
    assert start_from_name("clip.mkv") is None
    assert start_from_name("B8A44FD52B7F_20261399_250000.mkv") is None  # not a date


def test_probe(clip) -> None:
    path, frames = clip
    info = probe(path)
    assert (info.width, info.height, info.fps, info.frames) == (*SIZE, FPS, len(frames))
    with pytest.raises(OSError):
        probe(path.with_name("missing.mp4"))


def test_sampling_step() -> None:
    info = VideoInfo(path=None, width=96, height=64, fps=25.0, frames=90_000)
    assert sampling_step(info, 1.0, 90_000, 0) == 25
    assert sampling_step(info, 0.0, 90_000, 0) == 1  # every frame
    assert sampling_step(info, 1.0, 90_000, 1000) == 90  # stretched to at most 1000 samples
    assert sampling_step(info, 1.0, 250, 1000) == 25


def test_sampled_frames_are_every_step_th_frame_of_the_first_limit(clip) -> None:
    path, frames = clip
    stream = SampledFrames(path, step=7, limit=50, size=SIZE, prefetch=2)
    got = list(stream)
    stream.close()
    assert [i for i, _, _ in got] == list(range(0, 50, 7))
    assert [t for _, t, _ in got] == pytest.approx([i / FPS for i in range(0, 50, 7)])
    for i, _, frame in got:
        np.testing.assert_array_equal(frame, frames[i])
    assert stream.decoded == 50


def test_sampled_frames_are_fitted_to_the_model_geometry(clip) -> None:
    path, _ = clip
    stream = SampledFrames(path, step=30, limit=0, size=(48, 32))
    assert [f.shape for _, _, f in stream] == [(32, 48, 3)] * 2
    stream.close()


def test_a_frame_is_found_again_by_its_fingerprint(clip) -> None:
    path, frames = clip
    for index in (0, 3, 31, 59):
        found = find_frame(path, index, fingerprint(frames[index]), SIZE)
        np.testing.assert_array_equal(found, frames[index])
    assert find_frame(path, 10, np.zeros_like(fingerprint(frames[0])), SIZE) is None


class _OverSeeking:
    """A capture whose seeks land 20 frames past the target, reporting nothing."""

    def __init__(self, cap):
        self._cap = cap

    def set(self, prop, value):
        return self._cap.set(prop, value + 20)

    def __getattr__(self, name):
        return getattr(self._cap, name)


def test_a_bad_seek_falls_back_to_decoding_from_the_start(clip, monkeypatch) -> None:
    path, frames = clip
    real_open = reader._open
    monkeypatch.setattr(reader, "_SEEK_BACK", 4)
    monkeypatch.setattr(reader, "_open", lambda p: _OverSeeking(real_open(p)))
    found = find_frame(path, 30, fingerprint(frames[30]), SIZE)
    np.testing.assert_array_equal(found, frames[30])


def test_a_frame_keeps_the_time_the_file_stamps_on_it_across_a_hole(tmp_path) -> None:
    """Farm recordings lose stretches of video: 1 s, a 10 s hole, 1 s, in one file."""
    av = pytest.importorskip("av")
    from fractions import Fraction

    path = tmp_path / "20260330_072004_F923.mkv"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        for k, ms in enumerate([*range(0, 1000, 100), *range(11000, 12000, 100)]):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), k * 10, np.uint8), format="rgb24")
            frame.pts, frame.time_base = ms, Fraction(1, 1000)
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    stream = SampledFrames(path, step=5, limit=0, size=(64, 48))
    got = [(i, round(t, 2)) for i, t, _ in stream]
    stream.close()
    assert got == [(0, 0.0), (5, 0.5), (10, 11.0), (15, 11.5)]


def test_frames_are_found_by_decoding_forward(clip) -> None:
    path, frames = clip
    got = reader.frames_at(path, [31, 0, 59, 31, 60], SIZE)
    assert sorted(got) == [0, 31, 59]  # each once; past the end left out
    for i, frame in got.items():
        np.testing.assert_array_equal(frame, frames[i])
