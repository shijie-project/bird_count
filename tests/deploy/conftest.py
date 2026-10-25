import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdcount.deploy.shm import FrameRing, RingLayout


W, H = 96, 64


def make_layout(streams: int = 2, buffers: int = 4, size=(W, H)) -> RingLayout:
    return RingLayout(
        prefix=f"bc_test_{uuid.uuid4().hex[:8]}", num_streams=streams, num_buffers=buffers, frame_size=size
    )


@pytest.fixture
def ring():
    r = FrameRing.allocate(make_layout(), fps=10)
    yield r
    r.close()


def write_video(path: Path, frames: int, size=(W, H), fps: float = 25.0) -> Path:
    """MJPG .avi whose frame i is uniformly gray level 10 + 2*i (decodable everywhere)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), 10 + 2 * i, np.uint8))
    writer.release()
    return path
