from pathlib import Path

import numpy as np
import pytest


MAC = "B8A44FD52B7F"
SIZE = (96, 64)  # (width, height)
FPS = 10.0


@pytest.fixture(scope="module")
def clip(tmp_path_factory, make_video) -> tuple[Path, list[np.ndarray]]:
    path = tmp_path_factory.mktemp("video") / f"{MAC}_20260312_103023_ADA3.mp4"
    return path, make_video(path, 60, SIZE, FPS)
