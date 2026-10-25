"""DataLoaders for training crops and evaluation frames."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from birdcount.training.data.datasets import EvalFrames, TrainCrops
from birdcount.training.data.sampling import EpochSampler


def _collate_crops(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([s["image"] for s in batch]),  # (B, crop, crop, 3) uint8
        "density": torch.stack([s["density"] for s in batch]),  # (B, 1, crop/8, crop/8)
        "points": [s["points"] for s in batch],  # variable length
        "name": [s["name"] for s in batch],
    }


def _collate_frames(batch: list[dict]) -> dict:
    return {key: [s[key] for s in batch] for key in batch[0]}


def make_train_loader(
    dataset: TrainCrops, sampler: EpochSampler, *, batch_size: int, num_workers: int, pin_memory: bool = True
) -> DataLoader:
    """uint8 crops cross the worker boundary: a quarter of v2's float32 traffic."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_crops,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def make_eval_loader(dataset: EvalFrames, *, batch_size: int = 8, num_workers: int = 0) -> DataLoader:
    """Lists of whole frames, ready for `DensityPredictor.predict`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_frames,
        persistent_workers=False,
    )
