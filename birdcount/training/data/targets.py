"""Points on the density grid: the training target, and the cell each point falls in."""

from __future__ import annotations

import numpy as np


def point_cells(points: np.ndarray, stride: int, grid_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the density cell each continuous (x, y) point falls in.

    Cell (floor(y/stride), floor(x/stride)), clamped to the grid. The one
    point-to-cell rule shared by the training target and the evaluation
    region analyses.
    """
    h, w = grid_hw
    if len(points) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    cells = np.floor(np.asarray(points, dtype=np.float64) / stride).astype(np.int64)
    return np.clip(cells[:, 1], 0, h - 1), np.clip(cells[:, 0], 0, w - 1)


def count_map(points: np.ndarray, height: int, width: int, stride: int) -> np.ndarray:
    """(height/stride, width/stride) float32 map of how many points fall in each cell.

    The map's sum is exactly the number of points.
    """
    if height % stride or width % stride:
        raise ValueError(f"{width}x{height} is not divisible by stride {stride}")
    grid = (height // stride, width // stride)
    out = np.zeros(grid, dtype=np.float32)
    rows, cols = point_cells(points, stride, grid)
    np.add.at(out, (rows, cols), 1.0)
    return out
