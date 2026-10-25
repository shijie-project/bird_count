import os
import pickle

import cv2
import numpy as np

from birdcount.core.frames import fit_frame, read_image
from birdcount.training.data import cache as cache_mod
from birdcount.training.data.cache import FrameCache


def test_frames_are_fitted_and_sized(dataset_root, tmp_path) -> None:
    cache = FrameCache(tmp_path / "cache", (1280, 720))
    src = dataset_root / "images" / "train" / "a_1080p.png"
    entry = cache.get(src)
    assert entry.source_size == (1920, 1080)
    np.testing.assert_array_equal(entry.frame, fit_frame(read_image(src), (1280, 720)))
    assert not entry.frame.flags.writeable


def test_built_once_then_reused(dataset_root, tmp_path, monkeypatch) -> None:
    src = dataset_root / "images" / "train" / "b_720p.png"
    FrameCache(tmp_path / "cache", (1280, 720)).warm([src])
    calls = []
    monkeypatch.setattr(cache_mod, "read_image", lambda p: calls.append(p))
    FrameCache(tmp_path / "cache", (1280, 720)).get(src)  # a fresh instance, e.g. another worker
    assert calls == []


def test_changed_source_is_rebuilt(dataset_root, tmp_path) -> None:
    src = dataset_root / "images" / "train" / "b_720p.png"
    cache = FrameCache(tmp_path / "cache", (1280, 720))
    before = cache.get(src).frame.copy()
    data = read_image(src)
    data[:10] = 0
    cv2.imwrite(str(src), data)
    os.utime(src, ns=(1, 1))  # make sure the mtime moves
    after = FrameCache(tmp_path / "cache", (1280, 720)).get(src).frame
    assert (after[:10] == 0).all() and not (before[:10] == 0).all()


def test_pickles_without_open_handles(dataset_root, tmp_path) -> None:
    cache = FrameCache(tmp_path / "cache", (1280, 720))
    cache.get(dataset_root / "images" / "train" / "b_720p.png")
    clone = pickle.loads(pickle.dumps(cache))
    assert clone._open == {} and clone.dir == cache.dir
