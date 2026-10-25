"""The evaluation inference loop — one pass, shared by every analysis.

Frames come from `EvalFrames` (decoded once, at the deployment geometry) and
go through `core.inference.DensityPredictor`, the same pipeline as the live
runtime: fp16, region mask by camera MAC, density mask before the sum. So
evaluation measures the model as deployed, not a variant of it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from birdcount.core.camera_identity import extract_mac
from birdcount.core.inference import DensityPredictor
from birdcount.core.masks import RegionMasks
from birdcount.training.data.annotations import Sample
from birdcount.training.data.cache import FrameCache
from birdcount.training.data.datasets import EvalFrames
from birdcount.training.data.loader import make_eval_loader


@dataclass(frozen=True)
class ImageResult:
    name: str
    path: str
    frame: np.ndarray  # (H, W, 3) uint8 BGR, deployment geometry
    density: np.ndarray  # (H/8, W/8) float32, masked like deployment
    points: np.ndarray  # (N, 2) GT points in frame coordinates
    gt: float
    pred: float
    masked: bool
    gt_outside_mask: int  # GT points dropped because they lie in the ignored area


def masks_for(
    samples: Sequence[Sample], mask_dir: Path | None, frame_size: tuple[int, int]
) -> tuple[RegionMasks | None, list[int] | None]:
    """One mask row per camera (not per image), plus each image's row.

    Images whose name carries no known MAC map to an all-ones row (unmasked).
    """
    macs = [extract_mac(s.name) for s in samples]
    loaded = RegionMasks.from_dir(mask_dir, sorted({m for m in macs if m}), frame_size, name="eval")
    if loaded is None:
        return None, None
    width, height = frame_size
    keep = np.concatenate([loaded.keep, np.ones((1, height, width), np.uint8)])
    masks = RegionMasks(keep=keep, macs=(*loaded.macs, None))
    row_of = {mac: row for row, mac in loaded.covered.items()}
    unmasked = len(keep) - 1
    return masks, [row_of.get(mac, unmasked) for mac in macs]


def evaluate(
    samples: Sequence[Sample],
    predictor: DensityPredictor,
    cache: FrameCache,
    mask_ids: Sequence[int] | None,
    *,
    num_workers: int = 2,
) -> Iterator[ImageResult]:
    loader = make_eval_loader(EvalFrames(samples, cache), batch_size=predictor.max_batch, num_workers=num_workers)
    start = 0
    unmasked = None if predictor.masks is None else len(predictor.masks.keep) - 1
    for batch in loader:
        n = len(batch["image"])
        ids = None if mask_ids is None else list(mask_ids[start : start + n])
        pred = predictor.predict(batch["image"], ids, want_density=True)
        for i in range(n):
            masked = ids is not None and ids[i] != unmasked
            points = batch["points"][i]
            dropped = 0
            if masked:
                keep = predictor.masks.keep[ids[i]]
                h, w = keep.shape
                cols = np.clip(np.floor(points[:, 0]).astype(int), 0, w - 1)
                rows = np.clip(np.floor(points[:, 1]).astype(int), 0, h - 1)
                inside = keep[rows, cols].astype(bool)
                dropped = int((~inside).sum())
                points = points[inside]
            yield ImageResult(
                name=batch["name"][i],
                path=batch["path"][i],
                frame=batch["image"][i],
                density=pred.density[i].astype(np.float32),
                points=points,
                gt=float(len(points)),
                pred=float(pred.counts[i]),
                masked=masked,
                gt_outside_mask=dropped,
            )
        start += n
