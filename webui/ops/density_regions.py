"""Measure annotation errors inside connected blobs of a predicted density map.

This complements ``regional_density_error.py``: that operation uses a fixed
grid, while this one follows connected warm blobs in the prediction. For each
blob, prediction is the density integral and GT is the number of human points
mapped into that same blob. The overlay prints only the signed error, not an id.

Regions are the connected components of `density > --min-density` on the model's
native stride-8 grid — the same mask `utils.density_to_heatmap` colors, so a
region is exactly the warm area you already see in the overlays. A region's count
is the *sum of the density inside it*, which is the model's own count decomposed
by location: it is not a detection, and it is not integer. Two chickens standing
together are one region reading ~2.0, not two regions reading ~1.0 each.

The blob error is useful for finding prediction-shaped areas that over-count or
under-count. It is not a detection or a localization metric. See
`--min-count` / `--merge` for granularity.

Usage:
    # evaluate the validation split; output defaults beside the checkpoint
    python webui/ops/density_regions.py --data-path ../data/annotated --split val

    # coarser regions (bridge gaps up to 2 grid cells), ignore specks under 1 bird
    python webui/ops/density_regions.py --split all --merge 2 --min-count 1.0

    # match test.py's inference resolution
    python webui/ops/density_regions.py --split val --output-dir out --test-size 1280

The checkpoint comes from `--ckpt`, else MODEL_PATH in .env (same as test.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import cv2
import dotenv
import numpy as np
import torch
from PIL import Image


# Run as a plain script and sys.path[0] is this folder, so the project's own
# packages (datasets, models, utils) would not be importable — and `datasets`
# would resolve to the pip-installed HuggingFace one. Put the root first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.transforms import DOWNSAMPLE_RATIO  # noqa: E402
from models.shufflenet import get_shufflenet_density_model  # noqa: E402
from utils import HEATMAP_MIN_DENSITY, HEATMAP_VMAX, density_to_heatmap, set_seed  # noqa: E402
from webui.ops._evaluation_cli import (  # noqa: E402
    add_data_arguments,
    add_model_arguments,
    add_output_arguments,
    build_dataset,
    build_loader,
)


dotenv.load_dotenv()
warnings.simplefilter("ignore", UserWarning)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_ANNOTATIONS_JSON = PROJECT_ROOT.parent / "data" / "annotated" / "annotations" / "all.json"

# Overlay blend weights — same as test.py so these images read identically to
# the evaluation overlays you are already used to.
HEATMAP_ALPHA_BG = 0.5
HEATMAP_ALPHA_FG = 0.5

HEADER_COLOR = (0, 255, 255)  # BGR: per-image summary line
ERROR_OK_COLOR = (45, 170, 45)
ERROR_OVER_COLOR = (35, 35, 220)
ERROR_UNDER_COLOR = (220, 90, 25)


# ----------------------------------------------------------------------------
# Input discovery
# ----------------------------------------------------------------------------


def collect_images(paths: list[str], recursive: bool) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of image paths."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out.extend(q for q in it if q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            print(f"[warn] not found, skipped: {p}")
    return sorted(set(out))


def _image_key(value: str) -> str:
    """Match normal filenames and Label Studio's optional upload-hash prefix."""
    name = Path(value.replace("\\", "/")).name
    # Annotation image_id values are often already extensionless, while their
    # stems may legitimately contain dots (camera.mkv_timestamp or f0.50).
    # Path.stem would incorrectly truncate those names, so remove only a known
    # image extension.
    suffix = Path(name).suffix.casefold()
    stem = name[: -len(suffix)] if suffix in IMAGE_EXTS else name
    stem = stem.casefold()
    if len(stem) > 9 and stem[8] == "-" and all(char in "0123456789abcdef" for char in stem[:8]):
        stem = stem[9:]
    return stem


def load_annotation_points(path: str | Path) -> dict[str, np.ndarray]:
    """Load point annotations and index them by canonical image stem."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"annotation JSON not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    id_to_name = {
        str(image.get("id")): str(image.get("file_name", image.get("id")))
        for image in payload.get("images", [])
        if image.get("id") is not None
    }
    points_by_name: dict[str, np.ndarray] = {}
    for annotation in payload.get("annotations", []):
        image_id = str(annotation.get("image_id", ""))
        name = id_to_name.get(image_id, image_id)
        points = np.asarray(annotation.get("points") or [], dtype=np.float32).reshape(-1, 2)
        points_by_name[_image_key(name)] = points
    return points_by_name


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------


def resized_size(w: int, h: int, test_size: int) -> tuple[int, int]:
    """Size after `ResizeLongestEdge(test_size)`; replicated so we can find the
    unpadded extent of the output grid. `test_size <= 0` means native size."""
    long_edge = max(w, h)
    if test_size <= 0 or long_edge == test_size:
        return w, h
    scale = test_size / long_edge
    return max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)


@torch.inference_mode()
def predict_density(model, device, img_path: Path, transform, test_size: int) -> np.ndarray:
    """Run the model and return the (gh, gw) density map, padding cells removed.

    `PadToMultiple` appends right/bottom padding to make the image divisible by
    the stride; the model emits density over those cells too. Padded cells hold
    no chickens, so they are cropped off here — otherwise the padding's density
    would be attributed to whichever region touches the border.
    """
    with Image.open(img_path) as im:
        img = im.convert("RGB")
    w, h = img.size

    tensor, _ = transform(img, np.empty((0, 2)))
    outputs = model(tensor.unsqueeze(0).to(device).float())
    density = outputs[0, 0].float().cpu().numpy()

    w_r, h_r = resized_size(w, h, test_size)
    gh = min(density.shape[0], math.ceil(h_r / DOWNSAMPLE_RATIO))
    gw = min(density.shape[1], math.ceil(w_r / DOWNSAMPLE_RATIO))
    return density[:gh, :gw]


# ----------------------------------------------------------------------------
# Region extraction
# ----------------------------------------------------------------------------


def extract_regions(
    density: np.ndarray,
    min_density: float,
    merge: int,
    min_count: float,
) -> tuple[np.ndarray, list[dict]]:
    """Connect the density map into regions and integrate the count inside each.

    Args:
        density: (gh, gw) density in birds per grid cell.
        min_density: cells at or below this are background (blob threshold).
        merge: dilate the blob mask by this many cells before labeling, so blobs
            separated by a gap of up to `2 * merge` cells become one region.
            Counts still integrate the *original* density over the grown region,
            so dilation only re-absorbs sub-threshold mass — it never invents any.
        min_count: regions integrating fewer birds than this are dropped (their
            mass shows up in the reported residual instead).

    Returns:
        labels: (gh, gw) int32 label image, 0 = background, 1..N = kept regions
            renumbered so #1 is the most crowded.
        regions: per-region dicts, sorted by count descending.
    """
    mask = (density > min_density).astype(np.uint8)
    if merge > 0:
        k = 2 * merge + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    n_labels, raw_labels = cv2.connectedComponents(mask, connectivity=8)

    found = []
    for i in range(1, n_labels):  # 0 is background
        sel = raw_labels == i
        count = float(density[sel].sum())
        if count < min_count:
            continue
        found.append((count, i, int(sel.sum())))

    found.sort(key=lambda t: -t[0])

    labels = np.zeros_like(raw_labels, dtype=np.int32)
    regions: list[dict] = []
    for new_id, (count, old_id, area) in enumerate(found, start=1):
        sel = raw_labels == old_id
        labels[sel] = new_id
        ys, xs = np.nonzero(sel)
        # Density-weighted centroid: the label lands on the crowded part of the
        # region rather than the middle of an L-shaped blob.
        wts = density[sel]
        total = wts.sum()
        if total > 0:
            cx, cy = float((xs * wts).sum() / total), float((ys * wts).sum() / total)
        else:
            cx, cy = float(xs.mean()), float(ys.mean())
        regions.append(
            {
                "id": new_id,
                "count": round(count, 2),
                "count_rounded": int(round(count)),
                "area_cells": area,
                "_centroid_grid": (cx, cy),
                "_bbox_grid": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            }
        )
    return labels, regions


def extract_grid_regions(density: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, list[dict]]:
    """Split the frame into a fixed `rows` x `cols` tiling and sum each tile.

    The alternative to `extract_regions` for heavy pile-ups. Once birds are
    packed edge to edge, every blob touches its neighbour and the whole pile
    connects into one region reading e.g. 137.8 — true, but no more useful than
    the frame total. A fixed grid always decomposes: tiles are arbitrary, but
    "this cell of the grid holds ~9" is still something you can check by eye
    while placing points.

    Every tile is kept, including empty ones, and tiles are numbered row-major
    (left to right, top to bottom) so the ids match how you scan the image. The
    tile counts sum to the frame total exactly — no threshold, no residual.
    """
    gh, gw = density.shape
    y_edges = np.linspace(0, gh, rows + 1).round().astype(int)
    x_edges = np.linspace(0, gw, cols + 1).round().astype(int)

    labels = np.zeros((gh, gw), dtype=np.int32)
    regions: list[dict] = []
    for r in range(rows):
        y0, y1 = y_edges[r], y_edges[r + 1]
        for c in range(cols):
            x0, x1 = x_edges[c], x_edges[c + 1]
            tile_id = r * cols + c + 1
            labels[y0:y1, x0:x1] = tile_id
            count = float(density[y0:y1, x0:x1].sum())
            regions.append(
                {
                    "id": tile_id,
                    "row": r,
                    "col": c,
                    "count": round(count, 2),
                    "count_rounded": int(round(count)),
                    "area_cells": int((y1 - y0) * (x1 - x0)),
                    # Tile centre, not a density-weighted centroid: the number
                    # belongs to the tile as a whole, so anchoring it anywhere
                    # else would suggest a localization the count doesn't have.
                    "_centroid_grid": ((x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5),
                    "_bbox_grid": (int(x0), int(y0), int(x1 - 1), int(y1 - 1)),
                }
            )
    return labels, regions


def to_source_coords(regions: list[dict], gh: int, gw: int, width: int, height: int) -> None:
    """Attach source-image pixel geometry to each region, in place.

    The grid covers the whole image, so a cell maps to `width / gw` by
    `height / gh` source pixels. Coordinates come out in the same space as the
    `points` in the annotation JSONs, i.e. original image pixels.
    """
    sx, sy = width / gw, height / gh
    for r in regions:
        cx, cy = r.pop("_centroid_grid")
        x0, y0, x1, y1 = r.pop("_bbox_grid")
        r["centroid"] = [round((cx + 0.5) * sx, 1), round((cy + 0.5) * sy, 1)]
        # +1 on the far edge because the bbox is inclusive in cell indices.
        r["bbox"] = [
            round(x0 * sx, 1),
            round(y0 * sy, 1),
            round((x1 + 1) * sx - x0 * sx, 1),
            round((y1 + 1) * sy - y0 * sy, 1),
        ]


def attach_blob_errors(
    labels: np.ndarray,
    regions: list[dict],
    points: np.ndarray,
    *,
    width: int,
    height: int,
    point_snap: int = 0,
) -> int:
    """Attach GT and signed count error to each predicted blob.

    Points in label 0 are returned separately: those are likely missed areas
    and cannot honestly be assigned to a nearby predicted blob.
    """
    gh, gw = labels.shape
    if len(points):
        gx = np.clip((points[:, 0] * gw / width).astype(int), 0, gw - 1)
        gy = np.clip((points[:, 1] * gh / height).astype(int), 0, gh - 1)
        point_labels = labels[gy, gx].copy()
        # Density peaks and annotation points can differ by a cell or two even
        # for the same chicken. Snap only nearby background points; a point
        # farther from every blob remains an explicit missed area.
        for index in np.flatnonzero(point_labels == 0):
            x, y = int(gx[index]), int(gy[index])
            x0, x1 = max(0, x - point_snap), min(gw, x + point_snap + 1)
            y0, y1 = max(0, y - point_snap), min(gh, y + point_snap + 1)
            nearby_y, nearby_x = np.nonzero(labels[y0:y1, x0:x1])
            if not len(nearby_x):
                continue
            distances = np.square(nearby_x + x0 - x) + np.square(nearby_y + y0 - y)
            nearest = int(np.argmin(distances))
            point_labels[index] = labels[nearby_y[nearest] + y0, nearby_x[nearest] + x0]
    else:
        point_labels = np.empty(0, dtype=np.int32)

    counts = np.bincount(point_labels, minlength=len(regions) + 1)
    for region in regions:
        gt = int(counts[region["id"]])
        error = float(region["count"] - gt)
        region["gt_count"] = gt
        region["error"] = round(error, 3)
        region["abs_error"] = round(abs(error), 3)
        region["relative_error"] = round(error / gt * 100.0, 3) if gt else None
    return int(counts[0])


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------


def _draw_text(img, text: str, org: tuple[int, int], scale: float, color) -> None:
    """Text with a black halo so it stays readable over the heatmap."""
    thick = max(1, int(round(scale * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_overlay(
    original: np.ndarray,
    density: np.ndarray,
    labels: np.ndarray,
    regions: list[dict],
    vmax: float,
    gt_total: int,
    unassigned_gt: int,
    good_error: float,
    label_min_error: float,
) -> np.ndarray:
    """Draw signed errors on prediction-shaped blobs, with no visible ids."""
    H, W = original.shape[:2]
    density_full = cv2.resize(density, (W, H), interpolation=cv2.INTER_LINEAR)
    heat, mask = density_to_heatmap(density_full, vmax=vmax)

    overlay = original.copy()
    if mask.any():
        blended = cv2.addWeighted(original, HEATMAP_ALPHA_BG, heat, HEATMAP_ALPHA_FG, 0)
        cv2.copyTo(blended, mask, overlay)

    # cv2.resize has no int32 path; float32 round-trips these small ids exactly.
    labels_full = cv2.resize(labels.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    header_scale = max(0.5, min(1.4, H / 900.0))
    label_scale = max(0.35, min(1.0, H / 1800.0))
    outline_thickness = max(1, int(round(header_scale * 0.5)))

    tint = overlay.copy()
    for r in regions:
        error = r["error"]
        color = ERROR_OK_COLOR if abs(error) <= good_error else ERROR_OVER_COLOR if error > 0 else ERROR_UNDER_COLOR
        tint[labels_full == r["id"]] = color

    overlay = cv2.addWeighted(overlay, 0.82, tint, 0.18, 0)

    labeled = 0
    for r in regions:
        region_mask = (labels_full == r["id"]).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        error = r["error"]
        color = ERROR_OK_COLOR if abs(error) <= good_error else ERROR_OVER_COLOR if error > 0 else ERROR_UNDER_COLOR
        cv2.drawContours(overlay, contours, -1, color, outline_thickness + 1)
        if abs(error) < label_min_error:
            continue
        labeled += 1
        cx, cy = r["centroid"]
        _draw_text(overlay, f"{error:+.1f}", (int(cx), int(cy)), label_scale, color)

    errors = [region["error"] for region in regions]
    blob_mae = float(np.mean(np.abs(errors))) if errors else 0.0
    worst = float(max(np.abs(errors))) if errors else 0.0

    lines = [
        f"GT: {gt_total}   Pred: {density.sum():.1f}   Blob MAE: {blob_mae:.2f}   Worst: {worst:.2f}",
        f"GT outside blobs: {unassigned_gt}   Labeled errors: {labeled}/{len(regions)}",
    ]
    x0, y0 = int(H * 0.04), int(H * 0.08)
    for i, line in enumerate(lines):
        _draw_text(
            overlay, line, (x0, y0 + i * int(header_scale * 34)), header_scale * (1.0 if i == 0 else 0.6), HEADER_COLOR
        )
    return overlay


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure density-count error inside connected prediction blobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_arguments(p)
    add_model_arguments(p)

    g = p.add_argument_group("regions")
    g.add_argument(
        "--min-density",
        type=float,
        default=HEATMAP_MIN_DENSITY,
        help="cells at or below this density are background; the default is the same threshold the heatmap uses, "
        "so regions match the warm areas in the overlays",
    )
    g.add_argument(
        "--merge",
        type=int,
        default=1,
        help="bridge gaps of up to 2*N grid cells so nearby blobs form one region. 0 keeps raw connected "
        "components (in sparse areas that is roughly one region per bird); raise it for fewer, coarser regions",
    )
    g.add_argument(
        "--min-count",
        type=float,
        default=0.5,
        help="drop regions integrating fewer birds than this; their mass is reported as residual",
    )
    g.add_argument(
        "--label-min-error",
        type=float,
        default=0.5,
        help="only print blobs whose absolute error reaches this value; 0 prints every blob",
    )
    g.add_argument(
        "--good-error",
        type=float,
        default=1.0,
        help="a blob is colored green when its absolute count error is at most this value",
    )
    g.add_argument(
        "--point-snap",
        type=int,
        default=2,
        help="assign a human point just outside a blob to the nearest blob within this many density cells",
    )
    g.add_argument(
        "--top-regions",
        "--top-blobs",
        dest="top_regions",
        type=int,
        default=100,
        help="number of worst regions printed into the WebUI table; JSON always contains every region",
    )
    g.add_argument("--vmax", type=float, default=HEATMAP_VMAX, help="density painted as full-scale red")

    add_output_arguments(
        p,
        output_help="artifact directory; defaults to <checkpoint-dir>/blob_density_errors",
        overlay_help="write regions.json only, without per-image blob error overlays",
        legacy_overlay_flags=("--no-overlay", "--no-overlays"),
    )

    return p


def resolve_checkpoint(args) -> str:
    path = args.ckpt or os.getenv("MODEL_PATH")
    if not path:
        raise SystemExit("No checkpoint specified. Pass --ckpt or set MODEL_PATH in .env.")
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    return path


def parse_grid(spec: str | None) -> tuple[int, int] | None:
    """'4x6' -> (4, 6). Returns None when no grid was requested."""
    if not spec:
        return None
    try:
        rows, cols = (int(v) for v in spec.lower().split("x", 1))
    except ValueError:
        raise SystemExit(f"--grid expects ROWSxCOLS, e.g. '4x6' (got {spec!r})") from None
    if rows < 1 or cols < 1:
        raise SystemExit(f"--grid needs positive dimensions (got {spec!r})")
    return rows, cols


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    if args.good_error < 0 or args.label_min_error < 0 or args.point_snap < 0 or args.top_regions < 0:
        raise SystemExit("--good-error, --label-min-error, --point-snap, and --top-regions must be >= 0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.strip())
    set_seed(args.seed)
    try:
        dataset = build_dataset(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not len(dataset):
        raise SystemExit("No images found.")

    checkpoint = Path(resolve_checkpoint(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_shufflenet_density_model(model_path=checkpoint, device=device, fuse=not args.no_fuse)
    loader = build_loader(args, dataset)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent / "blob_density_errors"
    output_dir.mkdir(parents=True, exist_ok=True)

    size_str = f"longer edge = {args.test_size}px" if args.test_size > 0 else "native resolution"
    print(f"\nEvaluating '{args.split}' split ({len(dataset)} images, {size_str}), device={device}")
    # Phrased to match test.py's "Writing density overlays to:" — the web UI
    # scrapes this line to find the gallery directory for the run.
    print(f"Writing blob error overlays to: {output_dir}")
    print("-" * 78)

    records = []
    for sample in loader:
        img_path = Path(sample["path"][0])
        original = cv2.imread(str(img_path))
        if original is None:
            print(f"[warn] unreadable, skipped: {img_path}")
            continue
        H, W = original.shape[:2]

        inputs = sample["image"].to(device, non_blocking=True).float()
        output = model(inputs)[0, 0].float().cpu().numpy()
        valid_w, valid_h = resized_size(W, H, args.test_size)
        gh = min(output.shape[0], math.ceil(valid_h / DOWNSAMPLE_RATIO))
        gw = min(output.shape[1], math.ceil(valid_w / DOWNSAMPLE_RATIO))
        density = output[:gh, :gw]
        total = float(density.sum())
        points = sample["keypoints"][0].numpy()

        labels, regions = extract_regions(density, args.min_density, args.merge, args.min_count)
        unassigned_gt = attach_blob_errors(
            labels,
            regions,
            points,
            width=valid_w,
            height=valid_h,
            point_snap=args.point_snap,
        )
        to_source_coords(regions, gh, gw, W, H)

        # Region counts never quite add up to the image total: the mass under
        # --min-density and inside dropped regions is real prediction that no
        # region claims. Report it so a big residual doesn't hide from you.
        assigned = sum(r["count"] for r in regions)
        residual = total - assigned

        if not args.no_density_map:
            overlay = render_overlay(
                original,
                density,
                labels,
                regions,
                args.vmax,
                len(points),
                unassigned_gt,
                args.good_error,
                args.label_min_error,
            )
            cv2.imwrite(str(output_dir / f"{img_path.stem}_regions.png"), overlay)

        errors = np.asarray([region["error"] for region in regions], dtype=float)
        blob_mae = float(np.mean(np.abs(errors))) if len(errors) else 0.0
        worst = float(np.max(np.abs(errors))) if len(errors) else 0.0

        records.append(
            {
                "file_name": img_path.name,
                "path": str(img_path),
                "width": W,
                "height": H,
                "grid": [gh, gw],
                "total_count": round(total, 2),
                "gt_count": int(len(points)),
                "error": round(total - len(points), 3),
                "blob_mae": round(blob_mae, 3),
                "worst_blob_error": round(worst, 3),
                "gt_outside_blobs": unassigned_gt,
                "assigned_count": round(assigned, 2),
                "residual_count": round(residual, 2),
                "regions": regions,
            }
        )

        # Blob regions are already count-ordered, grid tiles are in reading
        # order — sort here so the preview shows the crowded ones either way.
        print(
            f"  {img_path.name}: GT {len(points):.1f} Pred {total:.1f} "
            f"BlobMAE {blob_mae:.3f} Worst {worst:.3f} GTOutside {unassigned_gt}"
        )
    manifest = output_dir / "regions.json"
    manifest.write_text(
        json.dumps(
            {
                "params": {
                    "mode": "blob_errors",
                    "data_path": str(Path(args.data_path)),
                    "split": args.split,
                    "min_density": args.min_density,
                    "merge": args.merge,
                    "min_count": args.min_count,
                    "label_min_error": args.label_min_error,
                    "good_error": args.good_error,
                    "point_snap": args.point_snap,
                    "top_regions": args.top_regions,
                    "test_size": args.test_size,
                    "downsample_ratio": DOWNSAMPLE_RATIO,
                },
                "images": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("-" * 78)
    table_blobs = [(record["file_name"], region) for record in records for region in record["regions"]]
    for image_name, region in sorted(table_blobs, key=lambda item: -item[1]["abs_error"])[: args.top_regions]:
        relative = "n/a" if region["relative_error"] is None else f"{region['relative_error']:+.1f}%"
        print(
            f"  BLOB {image_name} b{region['id']}: GT {region['gt_count']} "
            f"Pred {region['count']:.3f} Err {region['error']:+.3f} Err% {relative}"
        )
    all_errors = [region["error"] for record in records for region in record["regions"]]
    blob_count = len(all_errors)
    mae = float(np.mean(np.abs(all_errors))) if blob_count else 0.0
    rmse = float(np.sqrt(np.mean(np.square(all_errors)))) if blob_count else 0.0
    bias = float(np.mean(all_errors)) if blob_count else 0.0
    outside = sum(record["gt_outside_blobs"] for record in records)
    print(
        f"BLOB SUMMARY | Images {len(records)} | Blobs {blob_count} | MAE {mae:.4f} | "
        f"RMSE {rmse:.4f} | Bias {bias:+.4f} | GTOutside {outside}"
    )
    print(f"Region manifest: {manifest}")


if __name__ == "__main__":
    main()
