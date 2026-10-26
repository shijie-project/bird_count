"""Per-image hard-example mining for the counting trainer.

Tracks how wrong the model is on each training image and turns that into
sampling weights, so images the model keeps getting wrong come up more often.

Two things shape the design:

*Relative*, not absolute error. Train counts span ~9 to ~218 birds, so
`|pred - gt|` is dominated by the crowded images no matter how well the model
does on them. Weighting on absolute error would degenerate into "always train
on the densest images" instead of "train on what the model gets wrong", so the
error is divided by the crop's own GT count (floored by `min_count`, which also
keeps near-empty crops from exploding the ratio).

*Smoothed* over visits. Training samples a random square crop each time, so a
single observation mixes "this image is hard" with "this crop happened to be
hard". `observe()` therefore feeds an EMA per image rather than overwriting.

Weights are clipped to `[1/clip, clip]` around the mean. That cap matters: the
highest-error images are also where mislabeled data lives, and uncapped mining
would happily spend the whole epoch fitting bad annotations.
"""

from __future__ import annotations

import numpy as np


class HardExampleTracker:
    """EMA of per-image relative count error -> sampling weights.

    Args:
        names: image stems, in dataset order — index `i` of the returned weight
            vector is `names[i]`, matching the training dataset's sample indices.
        gamma: weight exponent. 0 is uniform, 1 is proportional to relative
            error, >1 sharpens further.
        ema_decay: retention of the running error per image (higher = smoother).
        clip: max ratio between a weight and the uniform weight, both ways.
        min_count: floor on the GT count used as the error denominator.
    """

    def __init__(
        self,
        names,
        *,
        gamma: float = 1.0,
        ema_decay: float = 0.9,
        clip: float = 3.0,
        min_count: float = 5.0,
    ):
        names = list(names)
        if not names:
            raise ValueError("names must be non-empty")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if clip < 1.0:
            raise ValueError(f"clip must be >= 1.0, got {clip}")
        if min_count <= 0:
            raise ValueError(f"min_count must be > 0, got {min_count}")

        self.names = names
        self.gamma = float(gamma)
        self.ema_decay = float(ema_decay)
        self.clip = float(clip)
        self.min_count = float(min_count)

        self._index = {n: i for i, n in enumerate(names)}
        self._err = np.zeros(len(names), dtype=np.float64)
        self._seen = np.zeros(len(names), dtype=bool)

    def __len__(self) -> int:
        return len(self.names)

    def observe(self, names, errors, gt_counts) -> None:
        """Fold one batch of observations into the per-image EMA.

        `errors` are signed or absolute count residuals (`pred - gt`) and
        `gt_counts` the GT count of the *crop* that produced them; both are
        per-sample and aligned with `names`. Names not in the tracked set are
        ignored, so a mid-run dataset change degrades quietly.
        """
        errors = np.abs(np.asarray(errors, dtype=np.float64)).reshape(-1)
        gt_counts = np.asarray(gt_counts, dtype=np.float64).reshape(-1)
        rel = errors / np.maximum(gt_counts, self.min_count)

        for name, r in zip(names, rel, strict=True):
            i = self._index.get(name)
            if i is None:
                continue
            # First sighting seeds the EMA outright; blending against the 0.0
            # init would otherwise take several visits to shake off.
            self._err[i] = r if not self._seen[i] else self.ema_decay * self._err[i] + (1.0 - self.ema_decay) * r
            self._seen[i] = True

    def weights(self) -> np.ndarray:
        """Sampling weights, mean-normalized and clipped. Uniform until seen."""
        w = np.ones(len(self._err), dtype=np.float64)
        if not self._seen.any() or self.gamma == 0.0:
            return w

        mean = float(self._err[self._seen].mean())
        if mean <= 0.0:  # model is exact everywhere (or nothing but empty crops)
            return w

        ratio = self._err / mean
        ratio[~self._seen] = 1.0  # not visited yet -> neutral, not starved
        return np.clip(np.power(ratio, self.gamma), 1.0 / self.clip, self.clip)

    def hardest(self, k: int = 5) -> list[tuple[str, float]]:
        """The `k` highest-error images seen so far, as (name, relative error)."""
        seen = np.flatnonzero(self._seen)
        if seen.size == 0:
            return []
        order = seen[np.argsort(-self._err[seen])][:k]
        return [(self.names[i], float(self._err[i])) for i in order]

    def coverage(self) -> float:
        """Fraction of tracked images observed at least once."""
        return float(self._seen.mean())

    # ----- checkpointing -----------------------------------------------------

    def state_dict(self) -> dict:
        return {"names": self.names, "err": self._err.tolist(), "seen": self._seen.tolist()}

    def load_state_dict(self, state: dict) -> int:
        """Restore per-image errors by name. Returns how many were matched.

        Matching by name rather than position means adding or removing training
        images between runs costs only the entries that actually changed.
        """
        matched = 0
        for name, err, seen in zip(state["names"], state["err"], state["seen"], strict=True):
            i = self._index.get(name)
            if i is None or not seen:
                continue
            self._err[i] = err
            self._seen[i] = True
            matched += 1
        return matched
