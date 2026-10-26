"""Evaluation metrics for crowd-counting density models.

Two kinds of consumers in mind:
- **Standard ML metrics** (MAE, RMSE, NAE, MAPE, R², Pearson, Bias) for paper /
  technical reporting.
- **Audience-friendly summaries** for exhibition: how often we're within a
  reasonable tolerance, how well pile-ups are caught at a chosen threshold.

Two of these are framed as accuracies, because they answer the question a
non-specialist actually asks. `counting_accuracy` is the share of the whole
flock counted correctly (weighted by flock size, so a 2-bird miss on a 10-bird
image barely registers next to a 12-bird miss on a pile-up); `accuracy_grid`
is the share of *images* landing inside a tolerance band. MAPE weights every
image equally and so is dominated by the smallest flocks — keep it for
comparability with the counting literature, but don't lead with it.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class CountingMetrics:
    """Aggregate count-regression metrics over a dataset."""

    n_images: int
    total_gt: float
    mae: float  # mean absolute error
    abs_mean: float  # mean of |pred - gt| (same as MAE; reported explicitly)
    abs_var: float  # population variance of |pred - gt|
    rmse: float  # root mean squared error
    nae: float  # MAE normalized by mean GT
    counting_accuracy: float  # 1 - sum|pred - gt| / sum(gt); flock-size weighted (== 1 - NAE)
    bias: float  # mean signed error (pred - gt); + = over-counts
    # MAPE weights every image equally, so a 2-bird miss on a 10-bird image hurts
    # more than a 12-bird miss on a 200-bird pile-up. Kept for comparability, but
    # `counting_accuracy` is the size-weighted number to lead with.
    mape: float  # mean absolute percentage error (skips GT=0); %
    rel_mean: float  # mean relative error |pred - gt|/gt (skips GT=0); % (same as MAPE)
    rel_var: float  # population variance of the relative error; %^2
    r2: float  # coefficient of determination
    pearson: float  # Pearson correlation between pred and gt
    worst_abs_error: float
    best_abs_error: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StratifiedMetrics:
    """MAE / MAPE within a single count band, e.g. 'Low (1-10)'."""

    band: str
    count_range: tuple[float, float]
    n_images: int
    mae: float
    mape: Optional[float]  # None if no images in this band have GT > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["count_range"] = list(d["count_range"])
        return d


@dataclass
class PileupClassificationMetrics:
    """Threshold-based classification: pile-up = (count > threshold)."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else float("nan")

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


DEFAULT_BANDS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("Empty", (0, 0)),
    ("Low (1-10)", (1, 10)),
    ("Medium (11-50)", (11, 50)),
    ("High (51-100)", (51, 100)),
    ("Pile-up (>100)", (101, float("inf"))),
)


def compute_metrics(preds: Sequence[float], gts: Sequence[float]) -> CountingMetrics:
    """Compute standard counting metrics from per-image predictions and GTs."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    if len(p) != len(g) or len(p) == 0:
        raise ValueError("preds and gts must be non-empty and equal-length")

    err = p - g
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    abs_mean = mae  # mean of the absolute error, by definition
    abs_var = float(abs_err.var())  # population variance (ddof=0)
    rmse = float(np.sqrt((err**2).mean()))
    mean_gt = float(g.mean())
    nae = mae / mean_gt if mean_gt > 0 else float("nan")
    # Total miscount as a fraction of the total flock, expressed as accuracy.
    # Algebraically 1 - NAE (mae / mean_gt == sum|err| / sum(gt)), but this is the
    # form we report, so it gets its own name.
    counting_accuracy = 1.0 - nae
    bias = float(err.mean())

    nonzero = g > 0
    if nonzero.any():
        rel_pct = abs_err[nonzero] / g[nonzero] * 100  # per-image relative error, %
        mape = float(rel_pct.mean())
        rel_mean = mape  # mean relative error, by definition == MAPE
        rel_var = float(rel_pct.var())  # population variance (ddof=0)
    else:
        mape = rel_mean = rel_var = float("nan")

    ss_res = float(((p - g) ** 2).sum())
    ss_tot = float(((g - mean_gt) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if g.std() > 0 and p.std() > 0:
        pearson = float(np.corrcoef(p, g)[0, 1])
    else:
        pearson = float("nan")

    return CountingMetrics(
        n_images=len(p),
        total_gt=float(g.sum()),
        mae=mae,
        abs_mean=abs_mean,
        abs_var=abs_var,
        rmse=rmse,
        nae=nae,
        counting_accuracy=counting_accuracy,
        bias=bias,
        mape=mape,
        rel_mean=rel_mean,
        rel_var=rel_var,
        r2=r2,
        pearson=pearson,
        worst_abs_error=float(abs_err.max()),
        best_abs_error=float(abs_err.min()),
    )


def compute_stratified(
    preds: Sequence[float],
    gts: Sequence[float],
    bands: Sequence[tuple[str, tuple[float, float]]] = DEFAULT_BANDS,
) -> list:
    """Per-band MAE / MAPE so we can see where errors concentrate."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    out = []
    for label, (lo, hi) in bands:
        mask = (g >= lo) & (g <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append(StratifiedMetrics(label, (lo, hi), 0, float("nan"), None))
            continue
        ae = np.abs(p[mask] - g[mask])
        mae = float(ae.mean())
        nz = g[mask] > 0
        mape = float((ae[nz] / g[mask][nz]).mean() * 100) if nz.any() else None
        out.append(StratifiedMetrics(label, (lo, hi), n, mae, mape))
    return out


def compute_pileup_classification(
    preds: Sequence[float], gts: Sequence[float], threshold: float
) -> PileupClassificationMetrics:
    """Treat each image as positive iff count > threshold; compute confusion stats."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    pred_pos = p > threshold
    gt_pos = g > threshold
    return PileupClassificationMetrics(
        threshold=float(threshold),
        tp=int((pred_pos & gt_pos).sum()),
        fp=int((pred_pos & ~gt_pos).sum()),
        tn=int((~pred_pos & ~gt_pos).sum()),
        fn=int((~pred_pos & gt_pos).sum()),
    )


def fraction_within(
    preds: Sequence[float],
    gts: Sequence[float],
    *,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> float:
    """Fraction of images where |pred - gt| <= max(abs_tol, rel_tol * gt).

    Either tolerance may be None to disable that mode. If both are provided,
    the looser tolerance per-image wins.
    """
    if abs_tol is None and rel_tol is None:
        raise ValueError("Pass at least one of abs_tol or rel_tol")
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    err = np.abs(p - g)
    tol = np.zeros_like(err)
    if abs_tol is not None:
        tol = np.maximum(tol, float(abs_tol))
    if rel_tol is not None:
        tol = np.maximum(tol, float(rel_tol) * g)
    return float((err <= tol).mean())


def accuracy_grid(
    preds: Sequence[float],
    gts: Sequence[float],
    abs_tols: Sequence[float],
    rel_tols: Sequence[float],
) -> dict[tuple[float, float], float]:
    """Tolerance accuracy for every (abs_tol, rel_tol) pair.

    Each cell is `fraction_within(abs_tol=a, rel_tol=r)`: the fraction of images
    whose miscount lands inside `max(a, r * gt)`. The combined rule is what makes
    this fair across flock sizes — small flocks lean on the absolute floor (being
    2 off on 10 birds is not a real error), large ones on the relative band.

    A tolerance of 0 switches off that half of the rule, so the `a = 0` row is
    pure relative tolerance and the `r = 0` column is pure absolute tolerance.
    """
    return {
        (float(a), float(r)): fraction_within(preds, gts, abs_tol=a, rel_tol=r) for a in abs_tols for r in rel_tols
    }
