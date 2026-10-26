from pathlib import Path

import cv2
import numpy as np
import pytest


MAC = "B8A44FD52B7F"
SIZE = (96, 64)  # (width, height)
FPS = 10.0


def write_video(path: Path, frames: int, size: tuple[int, int] = SIZE, fps: float = FPS) -> list[np.ndarray]:
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


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> tuple[Path, list[np.ndarray]]:
    path = tmp_path_factory.mktemp("video") / f"{MAC}_20260312_103023_ADA3.mp4"
    return path, write_video(path, 60)
