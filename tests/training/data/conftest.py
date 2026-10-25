import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def make_split(root: Path, split: str, images: dict[str, tuple[np.ndarray, list]]) -> Path:
    """Write images/<split>/<name>.png and annotations/<split>.json under `root`."""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(exist_ok=True)
    anns = []
    for name, (img, points) in images.items():
        cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
        if points is not None:
            anns.append({"image_id": name, "points": points})
    (root / "annotations" / f"{split}.json").write_text(json.dumps({"images": [], "annotations": anns}))
    return root


@pytest.fixture
def dataset_root(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    big = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)  # 1080p, gets fitted to 720p
    small = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    return make_split(
        tmp_path / "data",
        "train",
        {
            "a_1080p": (big, [[960.0, 540.0], [10.5, 20.25]]),
            "b_720p": (small, [[100.0, 200.0]]),
            "c_unlabelled": (small, None),
        },
    )
