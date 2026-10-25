"""Evaluation outputs: files for machines, a summary for people.

Everything a run produces is a file in its output folder — `metrics.json`
(all aggregate results), `per_image.csv`, and per-region JSON / CSV when those
analyses ran. Tools (the web UI) read these files; the printed summary is for
the terminal only and carries no contract.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from birdcount.training.eval.metrics import (
    ABS_TOLS,
    PRIMARY_ABS_TOL,
    PRIMARY_REL_TOL,
    REL_TOLS,
    CountingMetrics,
    PileupClassificationMetrics,
    StratifiedMetrics,
    accuracy_grid,
    compute_metrics,
    compute_pileup_classification,
    compute_stratified,
    fraction_within,
    within_tolerance,
)


def _strict_json(obj):
    """NaN / inf -> None, so files are strict JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _strict_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict_json(v) for v in obj]
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_strict_json(payload), indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Image-level counts
# ----------------------------------------------------------------------------


def count_metrics(preds: np.ndarray, gts: np.ndarray, pileup_threshold: float) -> dict:
    metrics = compute_metrics(preds, gts)
    pileup = compute_pileup_classification(preds, gts, pileup_threshold)
    grid = accuracy_grid(preds, gts, ABS_TOLS, REL_TOLS)
    return {
        "overall": metrics,
        "primary_accuracy": fraction_within(preds, gts, abs_tol=PRIMARY_ABS_TOL, rel_tol=PRIMARY_REL_TOL),
        "grid": grid,
        "stratified": compute_stratified(preds, gts),
        "pileup": pileup,
    }


def count_metrics_json(m: dict) -> dict:
    return {
        "overall": m["overall"].to_dict(),
        "tolerance_accuracy": {
            "primary": {
                "abs_tol_birds": PRIMARY_ABS_TOL,
                "rel_tol_fraction": PRIMARY_REL_TOL,
                "accuracy": m["primary_accuracy"],
            },
            "grid": [
                {"abs_tol_birds": a, "rel_tol_fraction": r, "accuracy": m["grid"][(a, r)]}
                for a in ABS_TOLS
                for r in REL_TOLS
            ],
        },
        "stratified": [s.to_dict() for s in m["stratified"]],
        "pileup_detection": m["pileup"].to_dict(),
    }


def write_per_image_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "name",
                "gt",
                "pred",
                "error",
                "abs_error",
                "rel_error_pct",
                "within_primary_tol",
                "masked",
                "gt_outside_mask",
            ]
        )
        for r in rows:
            err = r["pred"] - r["gt"]
            w.writerow(
                [
                    r["name"],
                    f"{r['gt']:.4f}",
                    f"{r['pred']:.4f}",
                    f"{err:+.4f}",
                    f"{abs(err):.4f}",
                    f"{err / r['gt'] * 100:+.4f}" if r["gt"] > 0 else "",
                    int(within_tolerance(err, r["gt"])),
                    int(r["masked"]),
                    r["gt_outside_mask"],
                ]
            )


def print_count_summary(m: dict) -> None:
    metrics: CountingMetrics = m["overall"]
    pileup: PileupClassificationMetrics = m["pileup"]
    stratified: list[StratifiedMetrics] = m["stratified"]
    tol = f"within +/-{PRIMARY_ABS_TOL:g} birds or +/-{PRIMARY_REL_TOL * 100:g}%"

    print()
    print("=" * 64)
    print(f"  Images                : {metrics.n_images:,}   (total GT birds {metrics.total_gt:,.0f})")
    print(f"  Counting accuracy     : {metrics.counting_accuracy * 100:.1f}%  (share of all birds counted correctly)")
    print(f"  Per-image accuracy    : {m['primary_accuracy'] * 100:.1f}%  ({tol})")
    print(f"  Average miscount      : {metrics.mae:.2f} birds per image (RMSE {metrics.rmse:.2f})")
    print(f"  Bias                  : {metrics.bias:+.2f} ({'over' if metrics.bias > 0 else 'under'}-counts)")
    print(f"  R^2 / Pearson         : {metrics.r2:.4f} / {metrics.pearson:.4f}")
    print(f"  MAPE                  : {metrics.mape:.2f}%  (unweighted: small flocks dominate)")
    actual = pileup.tp + pileup.fn
    label = f"Pile-ups (> {pileup.threshold:g})"
    print(
        f"  {label:<22}: {pileup.tp}/{actual} detected, {pileup.fp} false alarms "
        f"(P {pileup.precision:.3f} R {pileup.recall:.3f} F1 {pileup.f1:.3f})"
    )
    print()
    print(f"    {'band':<20}{'N':>5}{'MAE':>10}{'MAPE':>10}")
    for s in stratified:
        mae = "n/a" if math.isnan(s.mae) else f"{s.mae:.3f}"
        mape = "n/a" if s.mape is None else f"{s.mape:.1f}%"
        print(f"    {s.band:<20}{s.n_images:>5}{mae:>10}{mape:>10}")
    print()
    print("    abs \\ rel " + "".join(f"{f'{r:.0%}' if r else 'none':>8}" for r in REL_TOLS))
    for a in ABS_TOLS:
        label = f"+/-{a:g}" if a else "none"
        print(f"    {label:<10}" + "".join(f"{m['grid'][(a, r)] * 100:>7.1f}%" for r in REL_TOLS))
    print("=" * 64)


# ----------------------------------------------------------------------------
# Region analyses
# ----------------------------------------------------------------------------


def region_summary(errors) -> dict:
    """MAE / RMSE / bias over region errors (any iterable of floats)."""
    errors = np.fromiter(errors, dtype=float)
    if not len(errors):
        return {"regions": 0, "mae": 0.0, "rmse": 0.0, "bias": 0.0}
    return {
        "regions": len(errors),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "bias": float(errors.mean()),
    }


def write_regions_csv(path: Path, per_image: list[dict]) -> None:
    fields = ["image", "id", "row", "col", "gt", "count", "error", "abs_error", "relative_error"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for image in per_image:
            for region in image["regions"]:
                w.writerow({"image": image["name"], **region})


def print_worst_regions(title: str, per_image: list[dict], top: int) -> None:
    ranked = sorted(
        ((image["name"], r) for image in per_image for r in image["regions"]), key=lambda t: -t[1]["abs_error"]
    )
    if not ranked or top <= 0:
        return
    print(f"\n  {title}: {min(top, len(ranked))} largest errors")
    for name, r in ranked[:top]:
        where = f"r{r['row']}c{r['col']}" if r.get("row") else f"#{r['id']}"
        rel = "n/a" if r["relative_error"] is None else f"{r['relative_error']:+.1f}%"
        print(f"    {name} {where}: GT {r['gt']}  Pred {r['count']:.2f}  Err {r['error']:+.2f}  ({rel})")
