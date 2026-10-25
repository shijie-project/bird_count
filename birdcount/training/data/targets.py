"""The density target the model is trained against."""

from __future__ import annotations

import numpy as np


def count_map(points: np.ndarray, height: int, width: int, stride: int) -> np.ndarray:
    """(height/stride, width/stride) float32 map of how many points fall in each cell.

    A point at continuous (x, y) belongs to cell (floor(y/stride), floor(x/stride)),
    clamped to the grid; the map's sum is exactly the number of points.
    """
    if height % stride or width % stride:
        raise ValueError(f"{width}x{height} is not divisible by stride {stride}")
    h, w = height // stride, width // stride
    out = np.zeros((h, w), dtype=np.float32)
    if len(points):
        cells = np.floor(np.asarray(points, dtype=np.float64) / stride).astype(np.int64)
        np.add.at(out, (np.clip(cells[:, 1], 0, h - 1), np.clip(cells[:, 0], 0, w - 1)), 1.0)
    return out
