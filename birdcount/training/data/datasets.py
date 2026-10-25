"""PyTorch datasets over a split.

TrainCrops   random augmented crops for training. Indexed by (index, seed)
             pairs from `EpochSampler`, so a sample's augmentation depends
             only on its seed — not on which worker drew it or how many
             workers there are. Returns uint8 pixels; colour augmentation and
             normalisation happen on the GPU (`photometric`).
EvalFrames   whole frames at the model geometry, exactly as deployment sees
             them (BGR uint8, `fit_frame`d), with points scaled to match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from birdcount.core.model import OUTPUT_STRIDE
from birdcount.training.data.annotations import Sample
from birdcount.training.data.cache import FrameCache
from birdcount.training.data.geometry import crop_transform, warp_image, warp_points
from birdcount.training.data.targets import count_map


@dataclass(frozen=True)
class CropConfig:
    crop: int = 512
    # Longest edge the frame is resized to before cropping, sampled per crop:
    # several values expose the model to a range of bird sizes.
    train_sizes: tuple[int, ...] = (1280,)
    # Zoom jitter on top of the train size.
    scale_range: tuple[float, float] = (0.85, 1.2)
    hflip_p: float = 0.5
    rot90: bool = True

    def __post_init__(self):
        if self.crop % OUTPUT_STRIDE:
            raise ValueError(f"crop {self.crop} must be a multiple of {OUTPUT_STRIDE}")


def _scaled_points(sample: Sample, frame_size: tuple[int, int], source_size: tuple[int, int]) -> np.ndarray:
    """Points moved from the original image onto the fitted frame."""
    sx = frame_size[0] / source_size[0]
    sy = frame_size[1] / source_size[1]
    return sample.points * np.array([sx, sy])


class TrainCrops(Dataset):
    def __init__(self, samples: Sequence[Sample], cache: FrameCache, config: CropConfig | None = None):
        self.samples = list(samples)
        self.cache = cache
        self.config = config or CropConfig()
        cache.warm(s.path for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: tuple[int, int]) -> dict:
        index, seed = item
        sample = self.samples[index]
        cfg = self.config
        rng = np.random.default_rng(seed)

        cached = self.cache.get(sample.path)
        frame = cached.frame
        frame_size = (frame.shape[1], frame.shape[0])
        size = cfg.train_sizes[int(rng.integers(len(cfg.train_sizes)))]
        scale = size / max(frame_size) * rng.uniform(*cfg.scale_range)

        m = crop_transform(rng, frame_size, cfg.crop, scale, hflip_p=cfg.hflip_p, rot90=cfg.rot90)
        image = warp_image(frame, m, cfg.crop)
        points = warp_points(_scaled_points(sample, frame_size, cached.source_size), m, cfg.crop)
        density = count_map(points, cfg.crop, cfg.crop, OUTPUT_STRIDE)
        return {
            "image": torch.from_numpy(image),  # (crop, crop, 3) uint8 BGR
            "density": torch.from_numpy(density)[None],  # (1, crop/8, crop/8) float32
            "points": torch.from_numpy(points),  # (N, 2) float32, crop coordinates
            "name": sample.name,
        }


class EvalFrames(Dataset):
    def __init__(self, samples: Sequence[Sample], cache: FrameCache):
        self.samples = list(samples)
        self.cache = cache
        cache.warm(s.path for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        cached = self.cache.get(sample.path)
        frame_size = (cached.frame.shape[1], cached.frame.shape[0])
        return {
            "image": np.array(cached.frame),  # (H, W, 3) uint8 BGR, a private copy
            "points": _scaled_points(sample, frame_size, cached.source_size).astype(np.float32),
            "count": float(sample.count),
            "name": sample.name,
            "path": str(sample.path),
        }
