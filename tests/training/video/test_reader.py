import numpy as np
import pytest

from birdcount.training.video import reader
from birdcount.training.video.reader import SampledFrames, VideoInfo, find_frame, fingerprint, probe, sampling_step

from .conftest import FPS, SIZE


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
    assert [i for i, _ in got] == list(range(0, 50, 7))
    for i, frame in got:
        np.testing.assert_array_equal(frame, frames[i])
    assert stream.decoded == 50


def test_sampled_frames_are_fitted_to_the_model_geometry(clip) -> None:
    path, _ = clip
    stream = SampledFrames(path, step=30, limit=0, size=(48, 32))
    assert [f.shape for _, f in stream] == [(32, 48, 3)] * 2
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
