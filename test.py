"""Evaluate the trained density model on a dataset split.

Reports both audience-friendly summaries (for exhibition) and standard
counting metrics (for performance evaluation). Always writes a per-image
count CSV; optionally writes per-image density-overlay PNGs and a metrics
JSON report.

Two headline accuracies, because "how accurate is it?" has two honest answers:
`counting_accuracy` (share of the whole flock counted correctly, weighted by
flock size) and the per-image tolerance accuracy (share of frames landing
inside max(abs, rel x GT)). MAPE is kept for comparability but weights every
image equally, which over-penalises the small flocks nobody cares about.
"""

import argparse
import csv
import json
import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import cv2
import dotenv
import numpy as np
import torch

from metrics import (
    CountingMetrics,
    PileupClassificationMetrics,
    accuracy_grid,
    compute_metrics,
    compute_pileup_classification,
    compute_stratified,
    fraction_within,
)
from models.shufflenet import get_shufflenet_density_model
from utils import HEATMAP_MIN_DENSITY, density_to_heatmap, set_seed
from webui.ops._evaluation_cli import (
    add_data_arguments,
    add_model_arguments,
    add_output_arguments,
    build_dataset,
    build_loader,
)


dotenv.load_dotenv()
warnings.simplefilter("ignore", UserWarning)


# Overlay-blend weights (the density→color logic itself lives in utils).
HEATMAP_ALPHA_BG = 0.5
HEATMAP_ALPHA_FG = 0.5

# Per-cluster count labels: any connected density blob whose integrated count
# is at least this many birds gets its local count drawn at its centroid.
CLUSTER_MIN_COUNT = 0.3
# A per-cluster prediction is flagged red when |pred - gt| exceeds
# max(this floor, rel_tol * gt) — i.e. off by more than ~1 bird and rel_tol.
CLUSTER_ERR_ABS_FLOOR = 1.0

# Tolerance grid reported in the "within tolerance" table (not CLI-configurable).
# A prediction counts as correct when |pred - gt| <= max(abs_tol, rel_tol * gt),
# so a 0 switches off that half of the rule: the `none` row is pure relative
# tolerance and the `none` column is pure absolute tolerance.
ABS_TOLS = (0.0, 5.0, 10.0, 15.0, 20.0)  # absolute: within +/- N chickens
REL_TOLS = (0.0, 0.05, 0.10, 0.15, 0.20)  # relative: within +/- N% of GT count

# The headline per-image accuracy, and the same rule the overlays use to flag a
# bad frame. Small flocks lean on the 5-bird floor (missing 2 of 10 birds is
# inside annotation noise, not a real failure), pile-ups on the 10% band.
PRIMARY_ABS_TOL = 5.0
PRIMARY_REL_TOL = 0.10


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Kept separate from `parse_args` so other tools (notably the web UI, which
    generates its form from this spec) can introspect the flags without running
    an evaluation.
    """
    p = argparse.ArgumentParser(
        description="Evaluate density model on a dataset split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_data_arguments(p)
    add_model_arguments(p)

    g = p.add_argument_group("metrics")
    g.add_argument(
        "--pileup-threshold",
        type=float,
        default=100.0,
        help="count threshold above which an image is considered a pile-up event",
    )
    add_output_arguments(
        p,
        output_help="directory for all evaluation artifacts; defaults to the checkpoint's directory",
        overlay_help="skip per-image density overlay PNGs",
        include_metrics_out=True,
    )

    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _resolve_checkpoint(args) -> str:
    path = args.ckpt or os.getenv("MODEL_PATH")
    if not path:
        raise SystemExit("No checkpoint specified. Pass --ckpt or set MODEL_PATH in .env.")
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    return path


def _format_rel_err(err: float, gt: float) -> str:
    """Signed relative error (err / gt) as a percentage string; 'n/a' when GT == 0."""
    if gt <= 0:
        return "n/a"
    return f"{err / gt * 100:+.1f}%"


def _within_tol(err: float, gt: float, abs_tol: float, rel_tol: float) -> bool:
    """True if |err| is within tolerance (same rule as the `fraction_within` headline)."""
    return abs(err) <= max(abs_tol, rel_tol * gt)


# BGR colors for the header text: green = within tolerance, red = large error.
_HEADER_COLOR_OK = (0, 180, 0)
_HEADER_COLOR_BAD = (0, 0, 255)


def _draw_text(overlay, text: str, org, font_scale: float, color=(255, 255, 255)) -> None:
    """Colored text with a black outline so it stays legible over any JET color."""
    for c, thick in (
        ((0, 0, 0), max(2, int(round(font_scale * 4)))),
        (color, max(1, int(round(font_scale * 2)))),
    ):
        cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, thick, cv2.LINE_AA)


def _label_cluster_counts(
    overlay,
    pred_map: np.ndarray,
    W: int,
    H: int,
    font_scale: float,
    gt_grid: Optional[np.ndarray] = None,
    rel_tol: float = 0.10,
) -> None:
    """Draw each density blob's predicted count vs GT at its centroid, for failure analysis.

    Clusters are the connected components of the same above-threshold mask that
    `density_to_heatmap` colors, computed on the native (stride-8) `pred_map` so
    each label's number is the true count contribution of that blob (bilinear
    upsampling would rescale the sum). Centroids are mapped from grid to source
    pixels via the per-axis stride.

    The label reads ``pred/GT`` where GT is the number of ground-truth points
    (`gt_grid`, in the same stride-8 grid coords) that fall inside the blob. The
    text turns red when this pile's error is large (see `CLUSTER_ERR_ABS_FLOOR`),
    so you can immediately see *which* pile the model got wrong — not just that
    the image total is off.
    """
    h, w = pred_map.shape[:2]
    blob = (pred_map > HEATMAP_MIN_DENSITY).astype(np.uint8)
    n_labels, labels, _stats, centroids = cv2.connectedComponentsWithStats(blob, connectivity=8)

    if gt_grid is None or len(gt_grid) == 0:
        gt_grid = np.empty((0, 2), dtype=np.int64)
    # Blob label under each GT point (columns are x, rows are y in the grid).
    gt_labels = labels[gt_grid[:, 1], gt_grid[:, 0]] if len(gt_grid) else np.empty(0, dtype=np.int64)

    sx, sy = W / w, H / h
    for i in range(1, n_labels):  # 0 is background
        local = float(pred_map[labels == i].sum())
        gt_local = int((gt_labels == i).sum())
        if local < CLUSTER_MIN_COUNT and gt_local == 0:
            continue
        large_err = abs(local - gt_local) > max(CLUSTER_ERR_ABS_FLOOR, rel_tol * gt_local)
        color = (0, 0, 255) if large_err else (255, 255, 255)
        cx, cy = centroids[i]
        _draw_text(overlay, f"{local:.1f}/{gt_local}", (int(cx * sx), int(cy * sy)), font_scale, color=color)


def _save_overlay(
    img_path: str,
    pred_map: np.ndarray,
    gt_count: float,
    pred_count: float,
    out_path: Path,
    gt_grid: np.ndarray,
    abs_tol: float,
    rel_tol: float,
) -> None:
    """Save the raw source image with the predicted heatmap blended on top + count label.

    The density map (stride-8 model output) is bilinearly upsampled to the
    source image's resolution *before* colorization — interpolating the
    colored heatmap would blend LUT entries and produce wrong intermediate
    hues. Density→color goes through `utils.density_to_heatmap`, the single
    source of truth shared with the runtime monitor and side-by-side viz.
    """
    original = cv2.imread(img_path)
    if original is None:
        return  # source image not reachable from this machine; skip silently

    H, W = original.shape[:2]
    pred_full = cv2.resize(pred_map, (W, H), interpolation=cv2.INTER_LINEAR)

    heat, mask = density_to_heatmap(pred_full)

    overlay = original.copy()
    if mask.any():
        blended = cv2.addWeighted(original, HEATMAP_ALPHA_BG, heat, HEATMAP_ALPHA_FG, 0)
        cv2.copyTo(blended, mask, overlay)

    font_scale = max(0.5, min(2.5, H / 600.0))

    # Per-blob count labels so you can see *where* the model predicts birds and
    # how many at each cluster (not just the global total below).
    _label_cluster_counts(overlay, pred_map, W, H, font_scale * 0.7, gt_grid=gt_grid, rel_tol=rel_tol)
    err = pred_count - gt_count
    # Flag the whole header red when this image's error exceeds tolerance,
    # so large-error predictions stand out at a glance; green otherwise.
    header_color = _HEADER_COLOR_OK if _within_tol(err, gt_count, abs_tol, rel_tol) else _HEADER_COLOR_BAD
    lines = [
        f"GT: {gt_count:.1f}  Pred: {pred_count:.1f}",
        f"Abs Err: {abs(err):.1f}  Rel Err: {_format_rel_err(err, gt_count)}",
    ]
    x0 = int(H * 0.04)
    y0 = int(H * 0.08) + int(8 * font_scale)
    line_h = int(font_scale * 34)
    for i, line in enumerate(lines):
        _draw_text(overlay, line, (x0, y0 + i * line_h), font_scale, color=header_color)
    cv2.imwrite(str(out_path), overlay)


@torch.inference_mode()
def run_eval(model, device, loader, out_dir: Optional[Path]):
    names, preds, gts = [], [], []
    for sample in loader:
        inputs = sample["image"].to(device, non_blocking=True).float()
        gt_count = sample["density"].sum().item()
        outputs = model(inputs)
        pred_count = outputs.sum().item()
        err = pred_count - gt_count

        name = sample["name"][0]
        path = sample["path"][0]
        names.append(name)
        preds.append(pred_count)
        gts.append(gt_count)
        print(
            f"  {name}: GT {gt_count:7.1f}  Pred {pred_count:7.1f}  "
            f"Err {err:+7.1f}  Err% {_format_rel_err(err, gt_count):>8}"
        )

        if out_dir is not None:
            # Map GT points (input-pixel coords) onto the stride-8 output grid so
            # each blob can be scored against the GT points that fall inside it.
            kp = sample["keypoints"][0].cpu().numpy()
            gh, gw = outputs.shape[-2:]
            ih, iw = inputs.shape[-2:]
            if len(kp):
                gx = np.clip((kp[:, 0] * gw / iw).astype(np.int64), 0, gw - 1)
                gy = np.clip((kp[:, 1] * gh / ih).astype(np.int64), 0, gh - 1)
                gt_grid = np.stack([gx, gy], axis=1)
            else:
                gt_grid = np.empty((0, 2), dtype=np.int64)
            _save_overlay(
                path,
                outputs[0, 0].cpu().numpy(),
                gt_count,
                pred_count,
                out_dir / f"{name}_density.png",
                gt_grid,
                PRIMARY_ABS_TOL,
                PRIMARY_REL_TOL,
            )

    return names, np.asarray(preds), np.asarray(gts)


def _abs_tol_label(a: float) -> str:
    return f"+/-{a:g}" if a else "none"


def _rel_tol_label(r: float) -> str:
    return f"{r * 100:g}%" if r else "none"


def _primary_tol_label() -> str:
    return f"within +/-{PRIMARY_ABS_TOL:g} birds or +/-{PRIMARY_REL_TOL * 100:g}%"


def _print_exhibition_summary(
    metrics: CountingMetrics,
    pileup: PileupClassificationMetrics,
    primary_acc: float,
) -> None:
    bias_word = "over-counts" if metrics.bias > 0 else "under-counts"
    bias_amount = abs(metrics.bias)

    print()
    print("=" * 64)
    print("                   EXHIBITION SUMMARY")
    print("=" * 64)
    print(f"  Images analyzed         : {metrics.n_images:,}")
    print(f"  Total chickens (GT)     : {metrics.total_gt:,.0f}")
    print(f"  Counting accuracy       : {metrics.counting_accuracy * 100:.1f}%  (of all chickens counted)")
    print(f"  Per-image accuracy      : {primary_acc * 100:.1f}%  ({_primary_tol_label()})")
    print(f"  Average miscount        : {metrics.mae:.1f} chickens per image")
    print(f"  Best image              : off by {metrics.best_abs_error:.2f}")
    print(f"  Worst image             : off by {metrics.worst_abs_error:.2f}")
    print(f"  System bias             : {bias_word} by {bias_amount:.2f} on average")

    print()
    actual = pileup.tp + pileup.fn
    print(f"  Pile-ups in dataset     : {actual} (events with > {pileup.threshold:g} chickens)")
    if actual > 0:
        caught_pct = pileup.tp / actual * 100
        print(f"  Pile-ups detected       : {pileup.tp} of {actual} ({caught_pct:.0f}%)")
    print(f"  False alarms            : {pileup.fp}")
    print("=" * 64)


def _print_accuracy_grid(grid: dict[tuple[float, float], float], primary_acc: float) -> None:
    """Print the tolerance-accuracy table: rows are absolute, columns relative tolerance."""
    print()
    print("  Tolerance accuracy -- images within max(abs, rel x GT):")
    # Built with .ljust rather than an f-string: a backslash inside an f-string
    # replacement field is a syntax error before Python 3.12.
    print("    " + "abs \\ rel".ljust(10) + "".join(f"{_rel_tol_label(r):>9}" for r in REL_TOLS))
    for a in ABS_TOLS:
        row = "".join(f"{grid[(a, r)] * 100:>8.1f}%" for r in REL_TOLS)
        print(f"    {_abs_tol_label(a):<10}{row}")
    print(f"    headline: {primary_acc * 100:.1f}%  ({_primary_tol_label()})")


def _print_technical_metrics(
    metrics: CountingMetrics,
    stratified,
    pileup: PileupClassificationMetrics,
    grid: dict[tuple[float, float], float],
    primary_acc: float,
) -> None:
    print()
    print("=" * 64)
    print("                   TECHNICAL METRICS")
    print("=" * 64)
    print(f"  N images       : {metrics.n_images}")
    print(f"  Total GT count : {metrics.total_gt:.1f}")
    print(f"  MAE            : {metrics.mae:.4f}")
    print(f"  RMSE           : {metrics.rmse:.4f}")
    print(f"  NAE (MAE/mean) : {metrics.nae:.4f}")
    print(f"  Counting acc.  : {metrics.counting_accuracy:.4f}  (= 1 - NAE = 1 - sum|err| / sum GT)")
    print(f"  MAPE           : {metrics.mape:.2f} %  (unweighted; small flocks dominate)")
    print(f"  RelErr mean    : {metrics.rel_mean:.2f} %  (= MAPE)")
    print(f"  RelErr var     : {metrics.rel_var:.2f} %^2 (population)")
    print(f"  RelErr std     : {math.sqrt(metrics.rel_var):.2f} %")
    print(f"  Bias (signed)  : {metrics.bias:+.4f}")
    print(f"  R^2            : {metrics.r2:.4f}")
    print(f"  Pearson r      : {metrics.pearson:.4f}")
    print(f"  |Error| mean   : {metrics.abs_mean:.4f}  (= MAE)")
    print(f"  |Error| var    : {metrics.abs_var:.4f}  (population)")
    print(f"  |Error| std    : {math.sqrt(metrics.abs_var):.4f}")
    print(f"  |Error| best   : {metrics.best_abs_error:.2f}")
    print(f"  |Error| worst  : {metrics.worst_abs_error:.2f}")

    print()
    print("  Stratified MAE by GT count band:")
    print(f"    {'Band':<22} {'N':>5} {'MAE':>10} {'MAPE':>10}")
    for s in stratified:
        mape_str = f"{s.mape:.1f} %" if s.mape is not None else "  n/a"
        mae_str = f"{s.mae:.3f}" if not math.isnan(s.mae) else "n/a"
        print(f"    {s.band:<22} {s.n_images:>5} {mae_str:>10} {mape_str:>10}")

    _print_accuracy_grid(grid, primary_acc)

    print()
    print(f"  Pile-up detection (threshold = {pileup.threshold:g}):")
    print(f"    TP {pileup.tp:>4}   FP {pileup.fp:>4}   FN {pileup.fn:>4}   TN {pileup.tn:>4}")
    print(f"    Precision : {pileup.precision:.4f}")
    print(f"    Recall    : {pileup.recall:.4f}")
    print(f"    F1        : {pileup.f1:.4f}")
    print(f"    Accuracy  : {pileup.accuracy:.4f}")
    print("=" * 64)


def _write_per_image_csv(path: Path, names: Sequence[str], preds: np.ndarray, gts: np.ndarray) -> None:
    """Dump the raw per-image counts.

    Without this the per-image numbers only ever reach stdout, so re-scoring at a
    different tolerance — or plotting pred-vs-GT — means re-running the model over
    the whole split. The file is a few KB; always write it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "gt", "pred", "error", "abs_error", "rel_error_pct", "within_primary_tol"])
        for name, pred, gt in zip(names, preds, gts):
            err = float(pred) - float(gt)
            writer.writerow(
                [
                    name,
                    f"{gt:.4f}",
                    f"{pred:.4f}",
                    f"{err:+.4f}",
                    f"{abs(err):.4f}",
                    f"{err / gt * 100:+.4f}" if gt > 0 else "",
                    int(_within_tol(err, gt, PRIMARY_ABS_TOL, PRIMARY_REL_TOL)),
                ]
            )
    print(f"Per-image counts written to: {path}")


def _scrub_nans(obj):
    """Recursively replace NaN/inf with None so the result is strict JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _scrub_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nans(v) for v in obj]
    return obj


def _write_metrics_json(
    path: Path,
    metrics: CountingMetrics,
    stratified,
    pileup: PileupClassificationMetrics,
    grid: dict[tuple[float, float], float],
    primary_acc: float,
) -> None:
    payload = {
        "overall": metrics.to_dict(),
        "tolerance_accuracy": {
            "primary": {
                "abs_tol_chickens": PRIMARY_ABS_TOL,
                "rel_tol_fraction": PRIMARY_REL_TOL,
                "accuracy": primary_acc,
            },
            # The abs_tol = 0 entries are pure relative tolerance, the
            # rel_tol = 0 entries pure absolute — the old fraction_within_abs /
            # fraction_within_rel tables are the edges of this grid.
            "grid": [
                {"abs_tol_chickens": a, "rel_tol_fraction": r, "accuracy": grid[(a, r)]}
                for a in ABS_TOLS
                for r in REL_TOLS
            ],
        },
        "stratified": [s.to_dict() for s in stratified],
        "pileup_detection": pileup.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_scrub_nans(payload), indent=2))
    print(f"\nMetrics report written to: {path}")


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.strip())
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint(args)
    model = get_shufflenet_density_model(model_path=ckpt_path, device=device, fuse=not args.no_fuse)

    # Everything this run writes lands under one root: the explicit --output-dir,
    # or the checkpoint's own directory so artifacts stay next to the weights.
    output_dir = Path(args.output_dir) if args.output_dir else Path(ckpt_path).parent

    out_dir: Optional[Path] = None
    if not args.no_density_map:
        out_dir = output_dir / "density_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing density overlays to: {out_dir}")

    try:
        dataset = build_dataset(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    loader = build_loader(args, dataset)

    size_str = f"longer edge = {args.test_size}px" if args.test_size > 0 else "native resolution"
    print(f"\nEvaluating on '{args.split}' split ({len(dataset)} images, {size_str})")
    print("-" * 64)
    names, preds, gts = run_eval(model, device, loader, out_dir)
    print("-" * 64)

    metrics = compute_metrics(preds, gts)
    stratified = compute_stratified(preds, gts)
    pileup = compute_pileup_classification(preds, gts, args.pileup_threshold)
    grid = accuracy_grid(preds, gts, ABS_TOLS, REL_TOLS)
    # Computed directly rather than looked up in the grid so the headline does not
    # depend on the primary tolerances happening to be grid points.
    primary_acc = fraction_within(preds, gts, abs_tol=PRIMARY_ABS_TOL, rel_tol=PRIMARY_REL_TOL)

    _print_exhibition_summary(metrics, pileup, primary_acc)
    _print_technical_metrics(metrics, stratified, pileup, grid, primary_acc)

    print()
    _write_per_image_csv(output_dir / "per_image.csv", names, preds, gts)

    if args.metrics_out:
        _write_metrics_json(
            output_dir / args.metrics_out,
            metrics,
            stratified,
            pileup,
            grid,
            primary_acc,
        )


if __name__ == "__main__":
    main()
