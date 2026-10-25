"""Split all.json (+ images/all/) into train/val at a given ratio.

Produces the layout BirdDataset expects (see bird_count/datasets/bird.py):
    annotations/train.json   annotations/val.json
    images/train/            images/val/

The split is over IMAGES (each image and its matching annotation go to the
same side), stratified by CAMERA and grouped by RECORDING SESSION:

  * camera  - the 12-hex-digit MAC in the filename. A few legacy batches carry
              no MAC and are mapped via _CAMERA_BY_PREFIX; anything else falls
              into one "unknown" camera.
  * session - the recording the frame was cut from (date_time of the source
              video). Frames of one session are near-duplicates, so a session
              goes wholly to train or wholly to val; otherwise val would score
              memorised scenes rather than generalisation.

Each camera independently sends ~val_ratio of its images to val (whole
sessions, never overshooting the quota) and always at least one session, so
every camera is represented in val (unless it has only one session, which
stays in train). Annotations are matched to images by
filename stem, which is how image_id is stored (see convert_ls_to_coco.py).
The split is deterministic for a fixed --seed so re-running is reproducible.

images/train/ and images/val/ are emptied first so files from an earlier split
never linger.

Usage:
    python split_train_val.py                 # 8:2, seed 0, copies images
    python split_train_val.py --val-ratio 0.1 --seed 42
    python split_train_val.py --move          # move instead of copy
"""

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


_MAC_RE = re.compile(r"[0-9A-F]{12}")
# 20260325_164628_F92A -> video start time + segment hash, or MAC_20260808_065008.
_SESSION_RES = (re.compile(r"(\d{8}_\d{6})_[0-9A-F]{4}"), re.compile(r"[0-9A-F]{12}_(\d{8}_\d{6})"))
# 20210404_08_73.jpg -> legacy frames grouped by date + hour.
_LEGACY_SESSION_RE = re.compile(r"(\d{8}_\d{2})_")
# MAC-less batches whose camera was identified by eye against videos/<MAC>.mkv.
_CAMERA_BY_PREFIX = {"20260311_": "B8A44FD51C3C"}
UNKNOWN_CAMERA = "unknown"


def load_all(path: str):
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)
    return coco.get("images", []), coco.get("annotations", [])


def camera_key(file_name: str) -> str:
    match = _MAC_RE.search(file_name)
    if match:
        return match.group(0)
    for prefix, camera in _CAMERA_BY_PREFIX.items():
        if file_name.startswith(prefix):
            return camera
    return UNKNOWN_CAMERA


def session_key(file_name: str) -> str:
    if camera_key(file_name) != UNKNOWN_CAMERA:
        for pattern in _SESSION_RES:
            match = pattern.search(file_name)
            if match:
                return match.group(1)
    match = _LEGACY_SESSION_RE.match(file_name)
    return match.group(1) if match else file_name


def split_images(images: list, val_ratio: float, seed: int) -> tuple[list, list, dict]:
    """Per-camera, session-grouped split. Returns (train, val, {camera: (n_train, n_val)})."""
    if not 0 <= val_ratio <= 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    groups = defaultdict(lambda: defaultdict(list))
    for image in images:
        name = str(image.get("file_name", ""))
        groups[camera_key(name)][session_key(name)].append(image)

    rng = random.Random(seed)
    train_images, val_images, per_camera = [], [], {}
    for camera in sorted(groups):
        sessions = list(groups[camera].values())
        rng.shuffle(sessions)
        total = sum(len(s) for s in sessions)
        quota = max(1, round(total * val_ratio))

        val = []
        for session in sessions:
            if len(val) + len(session) <= quota:
                val += session
        if not val and len(sessions) > 1:
            # Every session is bigger than the quota: still give val the smallest one.
            # A single-session camera stays in train rather than vanishing from it.
            val = min(sessions, key=len)

        val_ids = {id(image) for image in val}
        train = [image for session in sessions for image in session if id(image) not in val_ids]
        train_images += train
        val_images += val
        per_camera[camera] = (len(train), len(val))
    return train_images, val_images, per_camera


def write_split(out_path: str, images: list, ann_by_stem: dict):
    images = sorted(images, key=lambda im: im["id"])
    stems = [os.path.splitext(im["file_name"])[0] for im in images]
    annotations = [ann_by_stem[stem] for stem in stems if stem in ann_by_stem]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations}, f, indent=2, ensure_ascii=False)
    return len(annotations)


def place_images(images: list, src_dir: str, dst_dir: str, move: bool):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    op = shutil.move if move else shutil.copy2
    missing = []
    for im in images:
        src = os.path.join(src_dir, im["file_name"])
        if not os.path.exists(src):
            missing.append(im["file_name"])
            continue
        op(src, os.path.join(dst_dir, im["file_name"]))
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", default="../data", help="")
    parser.add_argument("--input", default="all.json", help="combined json (images + annotations)")
    parser.add_argument("--images-dir", default="images/all", help="folder holding all image files")
    parser.add_argument("--ann-out-dir", default="annotations", help="where train.json/val.json go")
    parser.add_argument("--img-out-dir", default="images", help="parent of the train/ val/ image folders")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="per-camera fraction of images for val (default 0.2 -> 8:2)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for a reproducible shuffle")
    parser.add_argument("--move", action="store_true", help="move image files instead of copying them")
    args = parser.parse_args()

    args.input = Path(args.root_dir) / args.input
    args.images_dir = Path(args.root_dir) / args.images_dir
    args.ann_out_dir = Path(args.root_dir) / args.ann_out_dir
    args.img_out_dir = Path(args.root_dir) / args.img_out_dir

    images, annotations = load_all(args.input)
    ann_by_stem = {str(a["image_id"]): a for a in annotations}

    try:
        train_images, val_images, per_camera = split_images(images, args.val_ratio, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    os.makedirs(args.ann_out_dir, exist_ok=True)
    n_train_ann = write_split(os.path.join(args.ann_out_dir, "train.json"), train_images, ann_by_stem)
    n_val_ann = write_split(os.path.join(args.ann_out_dir, "val.json"), val_images, ann_by_stem)

    train_missing = place_images(train_images, args.images_dir, os.path.join(args.img_out_dir, "train"), args.move)
    val_missing = place_images(val_images, args.images_dir, os.path.join(args.img_out_dir, "val"), args.move)

    print(f"train: {len(train_images)} images, {n_train_ann} annotations")
    print(f"val:   {len(val_images)} images, {n_val_ann} annotations")
    print(f"{'camera':14} train  val")
    for camera, (n_train, n_val) in per_camera.items():
        print(f"{camera:14} {n_train:5} {n_val:4}")
    missing = train_missing + val_missing
    if missing:
        print(f"WARNING: {len(missing)} image file(s) listed in json not found in {args.images_dir}:")
        for name in missing[:10]:
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
