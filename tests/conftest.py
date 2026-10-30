from pathlib import Path

import cv2
import numpy as np
import pytest


def write_video(path: Path, frames: int, size: tuple[int, int], fps: float) -> list[np.ndarray]:
    """A clip of distinct noise frames; returns the frames as decoded (the codec is lossy)."""
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded.append(frame)
    cap.release()
    assert len(decoded) == frames
    return decoded


@pytest.fixture(scope="session")
def make_video():
    """`write_video(path, frames, size, fps)`: a test clip of noise frames, returned as decoded."""
    return write_video
