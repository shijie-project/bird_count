"""Epoch-seeded sampling: which samples, in what order, with which augmentation seed.

The sampler runs in the main process and hands each draw its own seed, so
an epoch is fully determined by (seed, epoch) — independent of worker count
and scheduling — and changing it from the trainer (`set_epoch`,
`set_weights`) works even with persistent workers.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class EpochSampler(Sampler[tuple[int, int]]):
    """Yields (index, seed) pairs.

    Without weights: every index exactly once per epoch, shuffled. With
    weights (hard-example mining): `num_samples` draws with replacement,
    proportional to the weights, so a hard image can appear several times.
    """

    def __init__(self, size: int, *, seed: int, num_samples: int | None = None):
        if size <= 0:
            raise ValueError("cannot sample from an empty dataset")
        self.size = size
        self.seed = seed
        self.num_samples = num_samples or size
        self.epoch = 0
        self.weights: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_weights(self, weights: np.ndarray | None) -> None:
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (self.size,) or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("weights must be non-negative, one per sample, not all zero")
        self.weights = weights

    def __len__(self) -> int:
        return self.size if self.weights is None else self.num_samples

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        if self.weights is None:
            order = rng.permutation(self.size)
        else:
            order = rng.choice(self.size, size=self.num_samples, replace=True, p=self.weights / self.weights.sum())
        seeds = rng.integers(0, 2**63 - 1, size=len(order))
        return zip(order.tolist(), seeds.tolist(), strict=True)
