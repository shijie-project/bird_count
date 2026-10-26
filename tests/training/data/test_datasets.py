import numpy as np
import pytest

from birdcount.training.data.annotations import load_split
from birdcount.training.data.cache import FrameCache
from birdcount.training.data.datasets import CropConfig, EvalFrames, TrainCrops
from birdcount.training.data.loader import make_eval_loader, make_train_loader
from birdcount.training.data.sampling import EpochSampler


@pytest.fixture
def cache(tmp_path) -> FrameCache:
    return FrameCache(tmp_path / "cache", (1280, 720))


def test_train_crop_shapes_and_mass(dataset_root, cache) -> None:
    ds = TrainCrops(load_split(dataset_root, "train"), cache, CropConfig(crop=256))
    s = ds[(0, 123)]
    assert s["image"].shape == (256, 256, 3) and s["image"].dtype.is_floating_point is False
    assert s["density"].shape == (1, 32, 32)
    assert s["density"].sum() == len(s["points"])


def test_same_seed_same_crop(dataset_root, cache) -> None:
    ds = TrainCrops(load_split(dataset_root, "train"), cache, CropConfig(crop=256))
    a, b, c = ds[(1, 7)], ds[(1, 7)], ds[(1, 8)]
    assert (a["image"] == b["image"]).all() and (a["points"] == b["points"]).all()
    assert not (a["image"] == c["image"]).all()


def test_eval_frames_match_deployment_geometry(dataset_root, cache) -> None:
    ds = EvalFrames(load_split(dataset_root, "train"), cache)
    s = ds[0]  # the 1080p image
    assert s["image"].shape == (720, 1280, 3)
    np.testing.assert_allclose(s["points"], [[640.0, 360.0], [7.0, 13.5]])
    assert s["count"] == 2.0


def test_loaders_with_workers_are_reproducible(dataset_root, cache) -> None:
    ds = TrainCrops(load_split(dataset_root, "train"), cache, CropConfig(crop=256))

    def epoch(workers: int) -> list:
        sampler = EpochSampler(len(ds), seed=5)
        sampler.set_epoch(2)
        loader = make_train_loader(ds, sampler, batch_size=2, num_workers=workers, pin_memory=False)
        return [b["image"] for b in loader]

    serial, parallel = epoch(0), epoch(2)
    assert len(serial) == 2
    assert all((a == b).all() for a, b in zip(serial, parallel, strict=True))


def test_eval_loader_yields_frame_lists(dataset_root, cache) -> None:
    batches = list(make_eval_loader(EvalFrames(load_split(dataset_root, "train"), cache), batch_size=2))
    assert [len(b["image"]) for b in batches] == [2, 1]
