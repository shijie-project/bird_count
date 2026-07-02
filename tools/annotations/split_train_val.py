"""Split all.json (+ images/all/) into train/val at a given ratio.

Produces the layout BirdDataset expects (see bird_count/datasets/bird.py):
    annotations/train.json   annotations/val.json
    images/train/            images/val/

The split is over IMAGES (each image and its matching annotation go to the
same side). Annotations are matched to images by filename stem, which is how
image_id is stored (see convert_ls_to_coco.py). The split is deterministic for
a fixed --seed so re-running is reproducible.

Usage:
    python split_train_val.py                 # 8:2, seed 0, copies images
    python split_train_val.py --val-ratio 0.1 --seed 42
    python split_train_val.py --move          # move instead of copy
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def load_all(path: str):
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)
    return coco.get("images", []), coco.get("annotations", [])


def write_split(out_path: str, images: list, ann_by_stem: dict):
    stems = {os.path.splitext(im["file_name"])[0] for im in images}
    annotations = [a for a in ann_by_stem.values() if str(a["image_id"]) in stems]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations}, f, indent=2, ensure_ascii=False)
    return len(annotations)


def place_images(images: list, src_dir: str, dst_dir: str, move: bool):
    os.makedirs(dst_dir, exist_ok=True)
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
    parser.add_argument("--val-ratio", type=float, default=0.2, help="fraction of images for val (default 0.2 -> 8:2)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for a reproducible shuffle")
    parser.add_argument("--move", action="store_true", help="move image files instead of copying them")
    args = parser.parse_args()

    args.input = Path(args.root_dir) / args.input
    args.images_dir = Path(args.root_dir) / args.images_dir
    args.ann_out_dir = Path(args.root_dir) / args.ann_out_dir
    args.img_out_dir = Path(args.root_dir) / args.img_out_dir

    images, annotations = load_all(args.input)
    ann_by_stem = {str(a["image_id"]): a for a in annotations}

    # Shuffle a copy of the image list, then slice off the val portion.
    order = list(images)
    random.Random(args.seed).shuffle(order)
    n_val = round(len(order) * args.val_ratio)
    val_images = order[:n_val]
    train_images = order[n_val:]

    os.makedirs(args.ann_out_dir, exist_ok=True)
    n_train_ann = write_split(os.path.join(args.ann_out_dir, "train.json"), train_images, ann_by_stem)
    n_val_ann = write_split(os.path.join(args.ann_out_dir, "val.json"), val_images, ann_by_stem)

    train_missing = place_images(train_images, args.images_dir, os.path.join(args.img_out_dir, "train"), args.move)
    val_missing = place_images(val_images, args.images_dir, os.path.join(args.img_out_dir, "val"), args.move)

    print(f"train: {len(train_images)} images, {n_train_ann} annotations")
    print(f"val:   {len(val_images)} images, {n_val_ann} annotations")
    missing = train_missing + val_missing
    if missing:
        print(f"WARNING: {len(missing)} image file(s) listed in json not found in {args.images_dir}:")
        for name in missing[:10]:
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
