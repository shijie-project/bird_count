"""The model's input geometry, and reading images from disk.

Every frame the model sees passes through `fit_frame`, wherever it came from
(a camera, a video file, a still). The pile-up thresholds were calibrated on
frames resized exactly this way — 1280x720, INTER_LINEAR, aspect ratio not
preserved — so a second resize rule anywhere would skew counts against them.

numpy + cv2 only: the frame grabber imports this without loading torch.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


DEFAULT_FRAME_SIZE = (1280, 720)  # (width, height)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_SUFFIXES = (".mkv", ".mp4", ".avi", ".mov", ".ts")


def fit_frame(frame: np.ndarray, size: tuple[int, int] = DEFAULT_FRAME_SIZE) -> np.ndarray:
    """Resize a frame to the model geometry `size` = (width, height).

    Returns `frame` itself when it already has that size, so the common case
    costs nothing.
    """
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def read_image(path: str | os.PathLike[str], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread that also works for non-ASCII paths on Windows.

    Raises instead of returning None, so a bad path fails where it is read.
    """
    path = Path(path)
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError as e:
        raise FileNotFoundError(f"cannot read image {path}: {e}") from e
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    return img
