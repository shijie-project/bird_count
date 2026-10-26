import numpy as np
import pytest

from birdcount.training.data.sampling import EpochSampler


def test_uniform_epoch_is_a_permutation_with_seeds() -> None:
    s = EpochSampler(10, seed=1)
    draws = list(s)
    assert sorted(i for i, _ in draws) == list(range(10))
    assert len({seed for _, seed in draws}) == 10


def test_determined_by_seed_and_epoch() -> None:
    a, b = EpochSampler(10, seed=1), EpochSampler(10, seed=1)
    assert list(a) == list(b)
    b.set_epoch(1)
    assert list(a) != list(b)


def test_weighted_draws_follow_the_weights() -> None:
    s = EpochSampler(3, seed=0, num_samples=30_000)
    s.set_weights(np.array([1.0, 2.0, 7.0]))
    counts = np.bincount([i for i, _ in s], minlength=3) / 30_000
    np.testing.assert_allclose(counts, [0.1, 0.2, 0.7], atol=0.01)
    assert len(s) == 30_000
    s.set_weights(None)
    assert len(s) == 3


@pytest.mark.parametrize("weights", [[1.0, 1.0], [0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])
def test_rejects_bad_weights(weights) -> None:
    with pytest.raises(ValueError):
        EpochSampler(3, seed=0).set_weights(np.array(weights))
