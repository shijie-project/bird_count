import random

import numpy as np
import torch

from .bird import BirdDataset
from .transforms import DOWNSAMPLE_RATIO, IMAGENET_MEAN, IMAGENET_STD


__all__ = [
    "BirdDataset",
    "DOWNSAMPLE_RATIO",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "collate",
    "seed_worker",
]


_STACK_KEYS = ("image", "density")


def collate(batch):
    """Dict-aware DataLoader collate.

    Stacks fixed-shape tensors (`image`, `density`); keeps variable-length /
    string fields (`keypoints`, `path`, `name`) as plain lists. Train uses fixed
    crops so stacking always works; val runs at batch=1 so per-image size
    differences are fine.
    """
    out = {}
    for key in batch[0]:
        vals = [s[key] for s in batch]
        out[key] = torch.stack(vals, 0) if key in _STACK_KEYS else vals
    return out


def seed_worker(_worker_id):
    """DataLoader worker init: seeds python/numpy RNGs from PyTorch's per-worker seed.

    Pass as `worker_init_fn=seed_worker` to ensure augmentations are independent
    across workers. `torch.initial_seed()` already returns the seed PyTorch
    derived for this worker, so we just rebroadcast it to `random`/`numpy`.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
