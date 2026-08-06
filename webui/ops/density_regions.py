"""Cluster a predicted density map into regions and report each region's chicken count.

A pre-annotation aid: run the density model on un-annotated images, connect the
density map into blobs, and integrate the density inside each blob. You get
"region #3 has about 12 chickens" instead of one number for the whole frame, so
you can walk the image region by region while placing points by hand.

Regions are the connected components of `density > --min-density` on the model's
native stride-8 grid — the same mask `utils.density_to_heatmap` colors, so a
region is exactly the warm area you already see in the overlays. A region's count
is the *sum of the density inside it*, which is the model's own count decomposed
by location: it is not a detection, and it is not integer. Two chickens standing
together are one region reading ~2.0, not two regions reading ~1.0 each.

What the count is good for: knowing how many points to place in an area, and
noticing when you have placed far too few. What it is not good for: exact
positions, or as ground truth. See `--min-count` / `--merge` for granularity.

Usage:
    # every image in a folder, overlays + regions.json next to them
    python webui/ops/density_regions.py ../data/raw/images -o ../data/raw/regions

    # coarser regions (bridge gaps up to 2 grid cells), ignore specks under 1 bird
    python webui/ops/density_regions.py ../data/raw/images -o out --merge 2 --min-count 1.0

    # heavy pile-up: blobs all merge into one, so tile the frame instead
    python webui/ops/density_regions.py pileup.jpg -o out --grid 4x6

    # match test.py's inference resolution
    python webui/ops/density_regions.py img.jpg -o out --test-size 1280

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

from datasets.transforms import DOWNSAMPLE_RATIO, build_val_transform  # noqa: E402
from models.shufflenet import get_shufflenet_density_model  # noqa: E402
from utils import HEATMAP_MIN_DENSITY, HEATMAP_VMAX, density_to_heatmap  # noqa: E402


dotenv.load_dotenv()
warnings.simplefilter("ignore", UserWarning)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Overlay blend weights — same as test.py so these images read identically to
# the evaluation overlays you are already used to.
HEATMAP_ALPHA_BG = 0.5
HEATMAP_ALPHA_FG = 0.5

CONTOUR_COLOR = (255, 255, 255)  # BGR: region outline
LABEL_COLOR = (255, 255, 255)  # BGR: "#3: 12.4" text
HEADER_COLOR = (0, 255, 255)  # BGR: per-image summary line


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
    total: float,
    label_min_count: float,
) -> np.ndarray:
    """Heatmap + region outlines + per-region counts, drawn on the source image.

    The density is upsampled before colorization (interpolating the *colored*
    heatmap would blend LUT entries into hues that aren't on the colormap), and
    the label image is upsampled with nearest-neighbour so region boundaries stay
    exactly on cell edges.

    Every region is outlined, but only those at or above `label_min_count` get a
    number drawn. A frame holds ~100 regions and most of them are single birds
    reading ~1.0; labeling all of them buries the piles — the ones you actually
    need a number for — under overlapping text.
    """
    H, W = original.shape[:2]
    density_full = cv2.resize(density, (W, H), interpolation=cv2.INTER_LINEAR)
    heat, mask = density_to_heatmap(density_full, vmax=vmax)

    overlay = original.copy()
    if mask.any():
        blended = cv2.addWeighted(original, HEATMAP_ALPHA_BG, heat, HEATMAP_ALPHA_FG, 0)
        cv2.copyTo(blended, mask, overlay)

    # cv2.resize has no int32 path; float32 round-trips these small ids exactly.
    labels_full = cv2.resize(labels.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    header_scale = max(0.5, min(2.5, H / 600.0))
    label_scale = max(0.35, min(1.2, H / 1400.0))
    outline_thickness = max(1, int(round(header_scale * 0.5)))

    labeled = 0
    for r in regions:
        region_mask = (labels_full == r["id"]).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, CONTOUR_COLOR, outline_thickness)
        if r["count"] < label_min_count:
            continue
        labeled += 1
        cx, cy = r["centroid"]
        _draw_text(overlay, f"#{r['id']}:{r['count']:.1f}", (int(cx), int(cy)), label_scale, LABEL_COLOR)

    lines = [
        f"Total: {total:.1f}   Regions: {len(regions)}",
        f"labeled: {labeled} region(s) with >= {label_min_count:g} birds",
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
        description="Cluster predicted density into regions and report each region's chicken count",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("paths", nargs="+", help="image files and/or directories of images")
    p.add_argument("-o", "--output-dir", required=True, help="directory for overlay PNGs and regions.json")
    p.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    p.add_argument("--limit", type=int, default=0, help="only process the first N images (0 = all)")

    g = p.add_argument_group("model")
    g.add_argument("--ckpt", default=None, help="checkpoint path; defaults to MODEL_PATH from .env")
    g.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")
    g.add_argument("--no-fuse", action="store_true", help="skip Conv+BN fusion (debug only)")
    g.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="resize the longer edge to this many pixels before inference (0 = native resolution)",
    )

    g = p.add_argument_group("regions")
    g.add_argument(
        "--grid",
        default=None,
        metavar="ROWSxCOLS",
        help="ignore the density blobs and tile the frame instead, e.g. '4x6'. Use this on heavy pile-ups, where "
        "every bird touches its neighbour and connected components collapse into one region covering the whole "
        "flock. Tile counts sum to the frame total exactly, so --min-density / --merge / --min-count do not apply",
    )
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
        "--label-min-count",
        type=float,
        default=1.5,
        help="only draw a number on regions with at least this many birds (all regions are still outlined, "
        "and all of them are in regions.json). Lower it to see every region's count at the cost of clutter",
    )
    g.add_argument("--vmax", type=float, default=HEATMAP_VMAX, help="density painted as full-scale red")
    g.add_argument("--no-overlay", action="store_true", help="write only regions.json, skip the PNGs")

    g = p.add_argument_group("label studio")
    g.add_argument(
        "--send-to-label-studio",
        action="store_true",
        help="after generating regions, create pre-label boxes and import the images into the selected LS project",
    )
    g.add_argument("--project-id", type=int, default=4, help="destination Label Studio project")
    g.add_argument(
        "--image-prefix",
        default="annotated\\images\\all\\",
        help="path from Label Studio's LOCAL_FILES_DOCUMENT_ROOT to these images",
    )
    g.add_argument(
        "--ls-min-count",
        type=float,
        default=1.5,
        help="only send region boxes predicting at least this many chickens; 0 sends every box",
    )
    g.add_argument(
        "--label-studio-url",
        default=os.getenv("LABEL_STUDIO_URL") or f"http://localhost:{os.getenv('LS_PORT', '8080')}",
        help="Label Studio base URL; LABEL_STUDIO_API_KEY is read only from .env",
    )
    g.add_argument(
        "--keep-label-config",
        action="store_true",
        help="do not install the required keypoint + region-box labeling interface before import",
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


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.strip())
    grid = parse_grid(args.grid)

    images = collect_images(args.paths, args.recursive)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit("No images found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_shufflenet_density_model(model_path=resolve_checkpoint(args), device=device, fuse=not args.no_fuse)
    transform = build_val_transform(DOWNSAMPLE_RATIO, args.test_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size_str = f"longer edge = {args.test_size}px" if args.test_size > 0 else "native resolution"
    print(f"\n{len(images)} image(s), {size_str}, device={device}")
    # Phrased to match test.py's "Writing density overlays to:" — the web UI
    # scrapes this line to find the gallery directory for the run.
    print(f"Writing region overlays to: {output_dir}")
    print("-" * 78)

    records = []
    for img_path in images:
        original = cv2.imread(str(img_path))
        if original is None:
            print(f"[warn] unreadable, skipped: {img_path}")
            continue
        H, W = original.shape[:2]

        density = predict_density(model, device, img_path, transform, args.test_size)
        total = float(density.sum())

        if grid is not None:
            labels, regions = extract_grid_regions(density, *grid)
        else:
            labels, regions = extract_regions(density, args.min_density, args.merge, args.min_count)
        gh, gw = density.shape
        to_source_coords(regions, gh, gw, W, H)

        # Region counts never quite add up to the image total: the mass under
        # --min-density and inside dropped regions is real prediction that no
        # region claims. Report it so a big residual doesn't hide from you.
        assigned = sum(r["count"] for r in regions)
        residual = total - assigned

        if not args.no_overlay:
            overlay = render_overlay(original, density, labels, regions, args.vmax, total, args.label_min_count)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_regions.png"), overlay)

        records.append(
            {
                "file_name": img_path.name,
                "path": str(img_path),
                "width": W,
                "height": H,
                "grid": [gh, gw],
                "total_count": round(total, 2),
                "assigned_count": round(assigned, 2),
                "residual_count": round(residual, 2),
                "regions": regions,
            }
        )

        # Blob regions are already count-ordered, grid tiles are in reading
        # order — sort here so the preview shows the crowded ones either way.
        busiest = sorted(regions, key=lambda r: -r["count"])[:6]
        top = "  ".join(f"#{r['id']}:{r['count']:.1f}" for r in busiest)
        more = f" (+{len(regions) - 6} more)" if len(regions) > 6 else ""
        print(f"  {img_path.name}: total {total:7.1f}  {len(regions):3d} regions  residual {residual:5.1f}")
        if top:
            print(f"      {top}{more}")

    manifest = output_dir / "regions.json"
    manifest.write_text(
        json.dumps(
            {
                "params": {
                    "mode": "grid" if grid else "blobs",
                    "grid": list(grid) if grid else None,
                    "min_density": args.min_density,
                    "merge": args.merge,
                    "min_count": args.min_count,
                    "label_min_count": args.label_min_count,
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
    grand = sum(r["total_count"] for r in records)
    print(f"{len(records)} image(s), {sum(len(r['regions']) for r in records)} regions, {grand:.1f} chickens total")
    print(f"Region manifest: {manifest}")

    if args.send_to_label_studio:
        # `tools/annotations.py` shadows the sibling annotations directory as a
        # package, so import the converter by putting its directory first.
        annotations_dir = PROJECT_ROOT / "tools" / "annotations"
        sys.path.insert(0, str(annotations_dir))
        from regions_to_label_studio import convert, import_to_label_studio

        ls_json = output_dir / "regions_ls.json"
        print("-" * 78)
        print(f"Preparing Label Studio pre-labels: {ls_json}")
        tasks = convert(
            manifest,
            ls_json,
            image_prefix=args.image_prefix,
            min_count=args.ls_min_count,
            project_id=args.project_id,
        )
        try:
            import_to_label_studio(
                tasks,
                project_id=args.project_id,
                url=args.label_studio_url,
                update_config=not args.keep_label_config,
            )
        except Exception as exc:
            raise SystemExit(f"Label Studio import failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
