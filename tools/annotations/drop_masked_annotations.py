"""Drop keypoints that fall inside a masked (blacked-out) image region.

Context: the dataset images already have a region mask baked in — everything
outside the usable region is painted pure black. The original mask.png files are
gone, so we recover the mask from the image itself: a point is "in the mask" when
a small patch around it is solidly black.

Reads the combined json (images + annotations, as produced by
convert_ls_to_coco.py), loads each image at native resolution, and removes every
annotation point whose surrounding patch is black. Writes a cleaned json and
prints how many points were dropped per image and in total.

Why a PATCH and not a single pixel:
    A genuinely dark chicken / shadow pixel would false-trigger a single-pixel
    test. Masked regions are large solid-black areas, so requiring the whole
    (2*radius+1) x (2*radius+1) patch to be black avoids deleting real birds.

Usage:
    python drop_masked_annotations.py \
        --input ../../../data/new_annotated/annotations/all.json \
        --images-dir ../../../data/new_annotated/images/all \
        --output ../../../data/new_annotated/annotations/all_clean.json
    python drop_masked_annotations.py --input all.json --images-dir images/all --report-only
"""

import argparse
import json
import os

import numpy as np
from PIL import Image


def point_in_mask(arr: np.ndarray, x: float, y: float, radius: int, thresh: int) -> bool:
    """True if the (2*radius+1) square patch centred on (x, y) is entirely black.

    `arr` is an (H, W, 3) uint8 array. Coordinates are clamped to the image so
    points on the very edge still get a (possibly truncated) patch to test.
    """
    h, w = arr.shape[:2]
    xi = min(max(int(round(x)), 0), w - 1)
    yi = min(max(int(round(y)), 0), h - 1)
    x0, x1 = max(xi - radius, 0), min(xi + radius + 1, w)
    y0, y1 = max(yi - radius, 0), min(yi + radius + 1, h)
    patch = arr[y0:y1, x0:x1]
    # "black" = every channel of every pixel in the patch is <= thresh.
    return bool(patch.max() <= thresh)


def clean(coco: dict, images_dir: str, radius: int, thresh: int):
    # stem -> file_name, so we can locate each annotation's image on disk.
    fname_by_stem = {os.path.splitext(im["file_name"])[0]: im["file_name"] for im in coco.get("images", [])}

    total_before = total_after = 0
    dropped_per_image = []  # (stem, n_dropped, n_kept)
    missing = []

    for ann in coco.get("annotations", []):
        stem = str(ann["image_id"])
        pts = ann.get("points", []) or []
        total_before += len(pts)

        file_name = fname_by_stem.get(stem)
        img_path = os.path.join(images_dir, file_name) if file_name else None
        if not file_name or not os.path.exists(img_path):
            missing.append(stem)
            total_after += len(pts)  # can't check -> keep points untouched
            continue

        arr = np.asarray(Image.open(img_path).convert("RGB"))
        kept = [p for p in pts if not point_in_mask(arr, p[0], p[1], radius, thresh)]
        n_drop = len(pts) - len(kept)
        if n_drop:
            dropped_per_image.append((stem, n_drop, len(kept)))
        ann["points"] = kept
        total_after += len(kept)

    return coco, total_before, total_after, dropped_per_image, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="combined json (images + annotations)")
    parser.add_argument("--images-dir", required=True, help="folder holding the (masked) image files")
    parser.add_argument("--output", default=None, help="output json (default: <input stem>_clean.json)")
    parser.add_argument("--radius", type=int, default=1, help="patch half-size; 1 => 3x3 must be black (default 1)")
    parser.add_argument(
        "--black-thresh", type=int, default=10, help="max channel value still counted as black (0-255)"
    )
    parser.add_argument("--report-only", action="store_true", help="print what would be dropped, do not write output")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        coco = json.load(f)

    coco, before, after, per_image, missing = clean(coco, args.images_dir, args.radius, args.black_thresh)
    dropped = before - after

    print(f"Points: {before} -> {after}  ({dropped} dropped inside masked regions)")
    if per_image:
        print("Per-image drops:")
        for stem, n_drop, n_kept in sorted(per_image, key=lambda t: -t[1]):
            print(f"  {stem}: -{n_drop}  (kept {n_kept})")
    if missing:
        print(f"WARNING: {len(missing)} image(s) not found on disk; their points were kept unchecked:")
        for stem in missing[:10]:
            print(f"  - {stem}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if args.report_only:
        print("\n--report-only: no file written.")
        return

    out = args.output or (os.path.splitext(args.input)[0] + "_clean.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"\nWrote cleaned json: {out}")


if __name__ == "__main__":
    main()
