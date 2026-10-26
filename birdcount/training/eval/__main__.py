"""Evaluate a checkpoint on a dataset split.

    python -m birdcount.training.eval --split val                  # counts + metrics
    python -m birdcount.training.eval --split val --blobs          # + errors per warm blob
    python -m birdcount.training.eval --split all --grid 4x6       # + errors per grid tile

One inference pass feeds every analysis. Frames are evaluated exactly as the
live runtime sees them: fitted to the deployment geometry, fp16, region mask
by camera MAC (--no-masks to disable).

Outputs, in --output-dir (default OUTPUT_ROOT/eval/<checkpoint>/<split>-<time>):
    metrics.json     every aggregate result
    per_image.csv    GT, prediction and error per image
    blobs.json/.csv  with --blobs
    grid.json/.csv   with --grid
    overlays/        <name>_counts.png, _blobs.png, _grid.png (unless --no-overlays)
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from birdcount.core.frames import IMAGE_SUFFIXES
from birdcount.core.inference import DensityPredictor
from birdcount.core.log import setup_logging
from birdcount.core.model import OUTPUT_STRIDE, load_inference_model
from birdcount.core.settings import CoreSettings
from birdcount.core.visualize import HEATMAP_MIN_DENSITY
from birdcount.training.data.annotations import Sample, load_split
from birdcount.training.data.cache import FrameCache
from birdcount.training.eval import overlays, report
from birdcount.training.eval.regions import assign_points, blob_regions, grid_regions
from birdcount.training.eval.runner import ImageResult, evaluate, masks_for


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m birdcount.training.eval",
        description="Evaluate a density-model checkpoint on a dataset split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("data")
    g.add_argument("--split", default="val", help="annotations/<split>.json + images/<split>/ under DATA_ROOT")
    g.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    g.add_argument("--skip", nargs="*", default=[], metavar="NAME", help="images to leave out (extension optional)")
    g.add_argument("--skip-unannotated", action="store_true", help="leave out images without annotation points")

    g = p.add_argument_group("model")
    g.add_argument("--ckpt", type=Path, default=None, help="checkpoint; default MODEL_PATH")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--no-masks", action="store_true", help="skip the per-camera region masks")
    g.add_argument("--frame-size", default=None, metavar="WxH", help="experiments only; default IMAGE_WIDTH x HEIGHT")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--workers", type=int, default=2, help="frame-loading workers")

    g = p.add_argument_group("analyses")
    g.add_argument("--pileup-threshold", type=float, default=100.0, help="count above which an image is a pile-up")
    g.add_argument("--blobs", action="store_true", help="score each connected warm blob of the density map")
    g.add_argument("--min-density", type=float, default=HEATMAP_MIN_DENSITY, help="blob threshold, birds per cell")
    g.add_argument("--merge", type=int, default=1, help="bridge blob gaps up to 2*N cells")
    g.add_argument("--min-count", type=float, default=0.5, help="drop blobs integrating fewer birds")
    g.add_argument("--point-snap", type=int, default=2, help="snap GT points to a blob within N cells")
    g.add_argument("--grid", default=None, metavar="ROWSxCOLS", help="score a fixed grid of tiles, e.g. 4x6")
    g.add_argument("--good-error", type=float, default=1.0, help="region drawn green when |error| <= this")
    g.add_argument("--label-min-error", type=float, default=0.5, help="label regions with |error| >= this")
    g.add_argument("--top-regions", type=int, default=20, help="worst regions printed (the files hold all)")

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", type=Path, default=None)
    g.add_argument("--no-overlays", action="store_true", help="files only, no overlay images")
    return p


def _pair(text: str, what: str, example: str) -> tuple[int, int]:
    try:
        a, b = (int(v) for v in text.lower().split("x", 1))
    except ValueError:
        raise SystemExit(f"{what} expects AxB, e.g. {example} (got {text!r})") from None
    if a < 1 or b < 1:
        raise SystemExit(f"{what} needs positive numbers (got {text!r})")
    return a, b


def _bare_name(name: str) -> str:
    path = Path(name)
    return (path.stem if path.suffix.lower() in IMAGE_SUFFIXES else path.name).casefold()


def _select(samples: list[Sample], args: argparse.Namespace) -> list[Sample]:
    if args.skip:
        wanted = {_bare_name(n) for n in args.skip}
        if unmatched := wanted - {s.name.casefold() for s in samples}:
            print(f"warning: --skip names matching no image: {sorted(unmatched)}")
        samples = [s for s in samples if s.name.casefold() not in wanted]
    if args.skip_unannotated:
        samples = [s for s in samples if s.annotated]
    return samples[: args.limit] if args.limit else samples


class _Analyses:
    """Per-image analyses beyond the count, and their overlays."""

    def __init__(self, args: argparse.Namespace, frame_size: tuple[int, int], out_dir: Path, pool: ThreadPoolExecutor):
        self.args = args
        self.frame_size = frame_size
        self.grid = _pair(args.grid, "--grid", "4x6") if args.grid else None
        self.overlay_dir = None if args.no_overlays else out_dir / "overlays"
        if self.overlay_dir:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.pool = pool
        self.blob_images: list[dict] = []
        self.grid_images: list[dict] = []

    def _save(self, name: str, image: np.ndarray) -> None:
        self.pool.submit(cv2.imwrite, str(self.overlay_dir / name), image)

    def __call__(self, res: ImageResult) -> None:
        a = self.args
        if self.overlay_dir:
            labels, blobs = blob_regions(res.density, self.frame_size, min_density=HEATMAP_MIN_DENSITY)
            assign_points(labels, blobs, res.points, OUTPUT_STRIDE)
            self._save(
                f"{res.name}_counts.png",
                overlays.count_overlay(res.frame, res.density, blobs, gt=res.gt, pred=res.pred),
            )
        if a.blobs:
            labels, regions = blob_regions(
                res.density, self.frame_size, min_density=a.min_density, merge=a.merge, min_count=a.min_count
            )
            outside = assign_points(labels, regions, res.points, OUTPUT_STRIDE, snap=a.point_snap)
            summary = report.region_summary(r.error for r in regions)
            assigned = sum(r.count for r in regions)
            self.blob_images.append(
                {
                    "name": res.name,
                    "gt": res.gt,
                    "pred": res.pred,
                    "gt_outside_blobs": outside,
                    "assigned_count": assigned,
                    "residual_count": res.pred - assigned,
                    **summary,
                    "regions": [r.to_dict() for r in regions],
                }
            )
            if self.overlay_dir:
                header = [
                    f"GT: {res.gt:.0f}   Pred: {res.pred:.1f}   Blob MAE: {summary['mae']:.2f}",
                    f"GT outside blobs: {outside}   Blobs: {len(regions)}",
                ]
                img = overlays.region_overlay(
                    res.frame,
                    res.density,
                    labels,
                    regions,
                    good_error=a.good_error,
                    label_min_error=a.label_min_error,
                    header=header,
                )
                self._save(f"{res.name}_blobs.png", img)
        if self.grid:
            labels, regions = grid_regions(res.density, self.frame_size, *self.grid)
            assign_points(labels, regions, res.points, OUTPUT_STRIDE)
            summary = report.region_summary(r.error for r in regions)
            self.grid_images.append(
                {
                    "name": res.name,
                    "gt": res.gt,
                    "pred": res.pred,
                    **summary,
                    "regions": [r.to_dict() for r in regions],
                }
            )
            if self.overlay_dir:
                header = [f"GT: {res.gt:.0f}   Pred: {res.pred:.1f}   Tile MAE: {summary['mae']:.2f}"]
                img = overlays.region_overlay(
                    res.frame,
                    None,
                    labels,
                    regions,
                    good_error=a.good_error,
                    label_min_error=a.label_min_error,
                    header=header,
                )
                self._save(f"{res.name}_grid.png", img)

    def write(self, out_dir: Path, payload: dict) -> None:
        a = self.args
        if a.blobs:
            params = {k: getattr(a, k) for k in ("min_density", "merge", "min_count", "point_snap")}
            errors = (r["error"] for im in self.blob_images for r in im["regions"])
            payload["blobs"] = {"params": params, **report.region_summary(errors)}
            report.write_json(out_dir / "blobs.json", {"params": params, "images": self.blob_images})
            report.write_regions_csv(out_dir / "blobs.csv", self.blob_images)
        if self.grid:
            rows, cols = self.grid
            errors = (r["error"] for im in self.grid_images for r in im["regions"])
            payload["grid"] = {"rows": rows, "cols": cols, **report.region_summary(errors)}
            report.write_json(out_dir / "grid.json", {"rows": rows, "cols": cols, "images": self.grid_images})
            report.write_regions_csv(out_dir / "grid.csv", self.grid_images)

    def print(self, payload: dict) -> None:
        for key, title, images in (("blobs", "Blobs", self.blob_images), ("grid", "Tiles", self.grid_images)):
            if key in payload:
                s = payload[key]
                print(f"  {title}: {s['regions']}  MAE {s['mae']:.3f}  RMSE {s['rmse']:.3f}  Bias {s['bias']:+.3f}")
                report.print_worst_regions(title, images, self.args.top_regions)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = CoreSettings.load()
    setup_logging(settings.log_level)

    checkpoint = args.ckpt or settings.model_path
    if checkpoint is None:
        raise SystemExit("No checkpoint: pass --ckpt or set MODEL_PATH.")
    frame_size = _pair(args.frame_size, "--frame-size", "1280x720") if args.frame_size else settings.frame_size
    samples = _select(load_split(settings.data_root, args.split), args)
    if not samples:
        raise SystemExit("No images to evaluate.")
    out_dir = args.output_dir or (
        settings.output_root / "eval" / Path(checkpoint).stem / f"{args.split}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    masks, mask_ids = (None, None) if args.no_masks else masks_for(samples, settings.mask_dir, frame_size)
    masked = 0 if masks is None else sum(i != len(masks.keep) - 1 for i in mask_ids)
    model = load_inference_model(checkpoint, args.device, input_order="bgr")
    predictor = DensityPredictor(model, frame_size=frame_size, masks=masks, max_batch=args.batch_size)
    cache = FrameCache(settings.output_root / "cache", frame_size)
    print(
        f"Evaluating {Path(checkpoint).name} on '{args.split}': {len(samples)} images at "
        f"{frame_size[0]}x{frame_size[1]}, {masked} with region masks -> {out_dir}"
    )

    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:  # PNG encoding overlaps inference
        analyses = _Analyses(args, frame_size, out_dir, pool)
        for res in evaluate(samples, predictor, cache, mask_ids, num_workers=args.workers):
            rows.append(
                {
                    "name": res.name,
                    "gt": res.gt,
                    "pred": res.pred,
                    "masked": res.masked,
                    "gt_outside_mask": res.gt_outside_mask,
                }
            )
            print(f"  {res.name}: GT {res.gt:7.1f}  Pred {res.pred:7.1f}  Err {res.pred - res.gt:+7.1f}")
            analyses(res)
    elapsed = time.perf_counter() - t0

    preds, gts = np.array([r["pred"] for r in rows]), np.array([r["gt"] for r in rows])
    metrics = report.count_metrics(preds, gts, args.pileup_threshold)
    payload = {
        "run": {
            "checkpoint": str(checkpoint),
            "split": args.split,
            "images": len(rows),
            "masked_images": masked,
            "gt_points_outside_masks": sum(r["gt_outside_mask"] for r in rows),
            "frame_size": list(frame_size),
            "precision": str(next(model.parameters()).dtype).removeprefix("torch."),
            "seconds": round(elapsed, 2),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        **report.count_metrics_json(metrics),
    }
    report.write_per_image_csv(out_dir / "per_image.csv", rows)
    analyses.write(out_dir, payload)
    report.write_json(out_dir / "metrics.json", payload)

    report.print_count_summary(metrics)
    analyses.print(payload)
    print(f"\n{len(rows)} images in {elapsed:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
