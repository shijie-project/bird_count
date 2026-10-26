"""Density target generation."""

import numpy as np


def gen_downsampled_density(im_height, im_width, ratio, points):
    """Build a (h/ratio, w/ratio) float32 count map directly from keypoints.

    Each point lands in the downsampled cell `(round(y) // ratio, round(x) //
    ratio)`, clamped to the grid. Total mass equals the number of points, even
    when multiple points fall in the same cell. Equivalent to allocating a
    full-resolution count map and sum-pooling, but skips the full-resolution
    intermediate.
    """
    if im_height % ratio or im_width % ratio:
        raise ValueError(f"density shape {(im_height, im_width)} not divisible by {ratio}")
    h_out, w_out = im_height // ratio, im_width // ratio
    out = np.zeros((h_out, w_out), dtype=np.float32)
    if len(points) == 0:
        return out
    pts = np.asarray(points)
    px = np.clip(np.round(pts[:, 0]).astype(np.int64) // ratio, 0, w_out - 1)
    py = np.clip(np.round(pts[:, 1]).astype(np.int64) // ratio, 0, h_out - 1)
    np.add.at(out, (py, px), 1.0)
    return out
