"""Reading a dataset split from disk.

Layout under DATA_ROOT (written by the labelling tools):

    images/<split>/<name>.jpg
    annotations/<split>.json     {"images": [...], "annotations": [{"image_id": "<name>", "points": [[x, y], ...]}]}

Points are in the original image's pixel coordinates, continuous (a point at
x lies in pixel floor(x)). An image with no annotation entry has no points
and counts as GT 0 — callers that want to skip those filter on `annotated`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Sample:
    name: str  # file stem; the annotation key
    path: Path
    points: np.ndarray  # (N, 2) float64 (x, y), original image coordinates
    annotated: bool  # has an annotation entry with at least one point

    @property
    def count(self) -> int:
        return len(self.points)


def load_split(data_root: Path, split: str) -> list[Sample]:
    """All images of `split`, sorted by name, with their points."""
    ann_path = data_root / "annotations" / f"{split}.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"annotation file not found: {ann_path} (generate it with the split tool)")
    image_dir = data_root / "images" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image folder not found: {image_dir}")

    with ann_path.open(encoding="utf-8") as f:
        coco = json.load(f)
    points_by_name = {
        str(ann["image_id"]): np.asarray(ann.get("points") or [], dtype=np.float64).reshape(-1, 2)
        for ann in coco.get("annotations", [])
    }

    samples = []
    for path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        points = points_by_name.get(path.stem)
        samples.append(
            Sample(
                name=path.stem,
                path=path,
                points=points if points is not None else np.empty((0, 2)),
                annotated=points is not None and len(points) > 0,
            )
        )
    return samples
