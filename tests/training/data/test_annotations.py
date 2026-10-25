import numpy as np
import pytest

from birdcount.training.data.annotations import load_split


def test_load_split(dataset_root) -> None:
    samples = {s.name: s for s in load_split(dataset_root, "train")}
    assert list(samples) == ["a_1080p", "b_720p", "c_unlabelled"]
    np.testing.assert_array_equal(samples["a_1080p"].points, [[960.0, 540.0], [10.5, 20.25]])
    assert samples["a_1080p"].annotated and samples["a_1080p"].count == 2
    assert not samples["c_unlabelled"].annotated and samples["c_unlabelled"].count == 0


def test_missing_split_fails_with_a_hint(dataset_root) -> None:
    with pytest.raises(FileNotFoundError, match=r"val\.json"):
        load_split(dataset_root, "val")
