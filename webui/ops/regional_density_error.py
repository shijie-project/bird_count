"""Evaluate a density model with fixed-grid regional count errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import dotenv
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.shufflenet import get_shufflenet_density_model
from utils import set_seed
from webui.ops._evaluation_cli import (
    add_data_arguments,
    add_model_arguments,
    add_output_arguments,
    build_dataset,
    build_loader,
)
from webui.ops.density_regions import resized_size


dotenv.load_dotenv(ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure fixed-grid regional density count errors against human keypoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_arguments(parser)
    add_model_arguments(parser)

    regions = parser.add_argument_group("regions")
    regions.add_argument("--grid", default="4x6", help="fixed grid as ROWSxCOLS")
    regions.add_argument(
        "--good-error",
        type=float,
        default=1.0,
        help="a region is colored green when its absolute count error is at most this value",
    )
    regions.add_argument(
        "--label-min-error",
        type=float,
        default=0.5,
        help="only print overlay labels whose absolute error reaches this value; 0 labels every region",
    )
    regions.add_argument(
        "--top-regions",
        type=int,
        default=100,
        help="number of worst regions printed into the WebUI table (JSON/CSV always contain all regions)",
    )

    add_output_arguments(
        parser,
        output_help="artifact directory; defaults to <checkpoint-dir>/regional_density_errors",
        overlay_help="write JSON/CSV only, without per-image regional error overlays",
        legacy_overlay_flags=("--no-overlays",),
    )
    return parser


def parse_grid(value: str) -> tuple[int, int]:
    try:
        rows, cols = (int(part) for part in value.lower().split("x", 1))
    except ValueError:
        raise ValueError(f"--grid expects ROWSxCOLS, for example 4x6 (got {value!r})") from None
    if rows < 1 or cols < 1:
        raise ValueError("--grid rows and columns must be positive")
    return rows, cols


def regional_errors(
    density: np.ndarray,
    keypoints: np.ndarray,
    rows: int,
    cols: int,
    *,
    valid_width: int,
    valid_height: int,
) -> list[dict]:
    """Integrate native density cells and count GT points in identical tiles."""
    gh, gw = density.shape
    if rows > gh or cols > gw:
        raise ValueError(f"grid {rows}x{cols} is finer than density output {gh}x{gw}")
    y_edges = np.linspace(0, gh, rows + 1).round().astype(int)
    x_edges = np.linspace(0, gw, cols + 1).round().astype(int)

    if len(keypoints):
        gx = np.clip((keypoints[:, 0] * gw / valid_width).astype(int), 0, gw - 1)
        gy = np.clip((keypoints[:, 1] * gh / valid_height).astype(int), 0, gh - 1)
    else:
        gx = gy = np.empty(0, dtype=int)

    records = []
    for row in range(rows):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for col in range(cols):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            pred = float(density[y0:y1, x0:x1].sum())
            gt = int(((gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)).sum())
            error = pred - gt
            records.append(
                {
                    "row": row + 1,
                    "col": col + 1,
                    "gt": gt,
                    "pred": pred,
                    "error": error,
                    "abs_error": abs(error),
                    "relative_error": error / gt * 100.0 if gt else None,
                    "grid_bbox": [x0, y0, x1, y1],
                }
            )
    return records


def _draw_text(image: np.ndarray, text: str, x: int, y: int, scale: float) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def save_overlay(
    image_path: str,
    output_path: Path,
    regions: list[dict],
    *,
    grid_width: int,
    grid_height: int,
    good_error: float,
    label_min_error: float,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        return
    height, width = image.shape[:2]
    tint = image.copy()
    for region in regions:
        x0, y0, x1, y1 = region["grid_bbox"]
        px0, px1 = round(x0 / grid_width * width), round(x1 / grid_width * width)
        py0, py1 = round(y0 / grid_height * height), round(y1 / grid_height * height)
        error = region["error"]
        color = (45, 170, 45) if abs(error) <= good_error else (35, 35, 220) if error > 0 else (220, 90, 25)
        cv2.rectangle(tint, (px0, py0), (px1, py1), color, -1)
    image = cv2.addWeighted(image, 0.78, tint, 0.22, 0)

    font_scale = max(0.35, min(0.8, height / 1100.0))
    for region in regions:
        if region["abs_error"] < label_min_error:
            continue
        x0, y0, x1, y1 = region["grid_bbox"]
        px0, px1 = round(x0 / grid_width * width), round(x1 / grid_width * width)
        py0, py1 = round(y0 / grid_height * height), round(y1 / grid_height * height)
        cv2.rectangle(image, (px0, py0), (px1, py1), (235, 235, 235), 1)
        text = f"{region['gt']}/{region['pred']:.1f} ({region['error']:+.1f})"
        _draw_text(image, text, px0 + 5, py0 + max(15, int(22 * font_scale)), font_scale)
    cv2.imwrite(str(output_path), image)


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.ckpt or os.getenv("MODEL_PATH") or "")
    if not str(checkpoint) or not checkpoint.is_file():
        raise ValueError("checkpoint not found; pass --ckpt or set MODEL_PATH in .env")
    return checkpoint


def _scrub(record: dict) -> dict:
    return {key: (round(value, 6) if isinstance(value, float) else value) for key, value in record.items()}


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    rows, cols = parse_grid(args.grid)
    checkpoint = _resolve_checkpoint(args)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.strip())
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_shufflenet_density_model(checkpoint, device=device, fuse=not args.no_fuse)
    dataset = build_dataset(args)
    loader = build_loader(args, dataset)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent / "regional_density_errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing regional error overlays to: {output_dir}")
    print(f"Grid: {rows}x{cols}  Device: {device}  Images: {len(dataset)}")

    images = []
    all_regions = []
    for sample in loader:
        input_tensor = sample["image"].to(device, non_blocking=True).float()
        output = model(input_tensor)[0, 0].float().cpu().numpy()

        image_path = sample["path"][0]
        original = cv2.imread(image_path)
        if original is None:
            print(f"[warn] unreadable image: {image_path}")
            continue
        source_h, source_w = original.shape[:2]
        valid_w, valid_h = resized_size(source_w, source_h, args.test_size)
        gh = min(output.shape[0], math.ceil(valid_h / 8))
        gw = min(output.shape[1], math.ceil(valid_w / 8))
        density = output[:gh, :gw]
        points = sample["keypoints"][0].numpy()
        regions = regional_errors(
            density,
            points,
            rows,
            cols,
            valid_width=valid_w,
            valid_height=valid_h,
        )
        name = sample["name"][0]
        for region in regions:
            region["image"] = name
            all_regions.append(region)

        gt_total = float(len(points))
        pred_total = float(density.sum())
        region_mae = float(np.mean([region["abs_error"] for region in regions]))
        worst = float(max(region["abs_error"] for region in regions))
        images.append(
            {
                "name": name,
                "gt": gt_total,
                "pred": pred_total,
                "error": pred_total - gt_total,
                "region_mae": region_mae,
                "worst_region": worst,
            }
        )
        print(f"  {name}: GT {gt_total:.1f} Pred {pred_total:.1f} RegionMAE {region_mae:.3f} Worst {worst:.3f}")
        if not args.no_density_map:
            save_overlay(
                image_path,
                output_dir / f"{name}_regional_error.png",
                regions,
                grid_width=gw,
                grid_height=gh,
                good_error=args.good_error,
                label_min_error=args.label_min_error,
            )

    ranked = sorted(all_regions, key=lambda item: item["abs_error"], reverse=True)
    for region in ranked[: args.top_regions]:
        rel = "n/a" if region["relative_error"] is None else f"{region['relative_error']:+.1f}%"
        print(
            f"  REGION {region['image']} r{region['row']}c{region['col']}: "
            f"GT {region['gt']} Pred {region['pred']:.3f} Err {region['error']:+.3f} Err% {rel}"
        )

    errors = np.asarray([region["error"] for region in all_regions], dtype=float)
    summary = {
        "images": len(images),
        "regions": len(all_regions),
        "regional_mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
        "regional_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0,
        "regional_bias": float(np.mean(errors)) if len(errors) else 0.0,
        "grid": f"{rows}x{cols}",
    }
    print(
        "REGIONAL SUMMARY | "
        f"Images {summary['images']} | Regions {summary['regions']} | "
        f"MAE {summary['regional_mae']:.4f} | RMSE {summary['regional_rmse']:.4f} | "
        f"Bias {summary['regional_bias']:+.4f}"
    )

    payload = {"summary": summary, "images": images, "regions": [_scrub(region) for region in all_regions]}
    (output_dir / "regional_errors.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "regional_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image", "row", "col", "gt", "pred", "error", "abs_error", "relative_error"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["regions"])
    print(f"Regional JSON: {output_dir / 'regional_errors.json'}")
    print(f"Regional CSV: {output_dir / 'regional_errors.csv'}")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    if args.good_error < 0 or args.label_min_error < 0 or args.top_regions < 0:
        raise SystemExit("--good-error, --label-min-error, and --top-regions must be >= 0")
    try:
        evaluate(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Regional evaluation failed: {exc}") from exc


if __name__ == "__main__":
    main()
