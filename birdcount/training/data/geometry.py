"""Geometric augmentation as one affine transform.

Training crops are: rescale the frame, take a crop x crop window (padding
with black where the scaled frame is smaller), maybe mirror, rotate by a
multiple of 90 degrees. All of it is linear, so it composes into one 2x3
matrix and the crop is produced by a single `cv2.warpAffine` of crop x crop
output pixels — instead of resizing the whole frame first (v2 spent ~11 ms
per sample on full-frame resizes).

Coordinates are continuous throughout: pixel i covers [i, i+1), so a point
at x lies in pixel floor(x), and mirroring a width-w image maps x to w - x.
The same matrix moves the image and the points, so they cannot drift apart.
"""

from __future__ import annotations

import cv2
import numpy as np


def crop_transform(
    rng: np.random.Generator,
    frame_size: tuple[int, int],
    crop: int,
    scale: float,
    *,
    hflip_p: float = 0.5,
    rot90: bool = True,
) -> np.ndarray:
    """A random (3, 3) affine mapping frame coordinates to crop coordinates.

    The frame is scaled by `scale`; along each axis the crop window starts
    at a random integer offset — inside the scaled frame when it is larger
    than `crop`, else placing the frame at a random position on black.
    """
    m = np.diag([scale, scale, 1.0])
    for axis, length in enumerate(frame_size):
        span = round(length * scale) - crop
        offset = int(rng.integers(min(0, span), max(0, span) + 1))
        m[axis, 2] = -offset

    if rng.random() < hflip_p:
        m = np.array([[-1.0, 0, crop], [0, 1, 0], [0, 0, 1]]) @ m
    k = int(rng.integers(0, 4)) if rot90 else 0
    for _ in range(k):  # 90 degrees counter-clockwise: (x, y) -> (y, crop - x)
        m = np.array([[0.0, 1, 0], [-1, 0, crop], [0, 0, 1]]) @ m
    return m


def warp_image(image: np.ndarray, m: np.ndarray, crop: int) -> np.ndarray:
    """Apply `m` to an (H, W, C) image; out-of-frame pixels are black.

    Bicubic, not bilinear: the zoom jitter enlarges most crops, and bilinear
    enlargement softens the birds' edges and texture below what the cameras
    deliver. Val counting accuracy (as deployed) over the last 200 of 1000
    epochs: bilinear 92.9-93.7 % (3 seeds), v2's PIL resampling 94.3 %,
    bicubic 94.9-95.2 % (3 runs).
    """
    # cv2 works in pixel-index coordinates (centres at integers): with
    # u = x + 0.5, u' = A u + b becomes x' = A x + (A @ 0.5 + b - 0.5).
    a, b = m[:2, :2], m[:2, 2]
    idx = np.hstack([a, (a @ [0.5, 0.5] + b - 0.5)[:, None]])
    return cv2.warpAffine(
        image, idx, (crop, crop), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def warp_points(points: np.ndarray, m: np.ndarray, crop: int) -> np.ndarray:
    """Apply `m` to (N, 2) points and drop those that leave the crop."""
    if len(points) == 0:
        return np.empty((0, 2), np.float32)
    out = points @ m[:2, :2].T + m[:2, 2]
    inside = (out >= 0).all(axis=1) & (out < crop).all(axis=1)
    return out[inside].astype(np.float32)
