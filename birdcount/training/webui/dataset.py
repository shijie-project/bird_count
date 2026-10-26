"""The dataset as the browser page sees it: splits, their images, each image's points.

Same layout as `training.data.annotations` (images/<split>/, annotations/<split>.json),
read with the stdlib only: the panel never loads numpy, cv2 or torch. Points
are drawn by the page; predictions come from an evaluation run of the split.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from birdcount.core.paths import IMAGE_SUFFIXES


class Dataset:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._cache: dict[str, tuple[float, dict[str, list]]] = {}  # split -> (mtime, points by name)
        self._lock = threading.Lock()

    def splits(self) -> list[dict]:
        out = []
        for ann in sorted((self.data_root / "annotations").glob("*.json")):
            images = self.data_root / "images" / ann.stem
            if images.is_dir():
                out.append({"name": ann.stem, "images": sum(1 for p in images.iterdir() if _is_image(p))})
        return out

    def _points(self, split: str) -> dict[str, list]:
        _check(split)
        path = self.data_root / "annotations" / f"{split}.json"
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(split)
            if cached is None or cached[0] != mtime:
                data = json.loads(path.read_text(encoding="utf-8"))
                points = {str(a["image_id"]): a.get("points") or [] for a in data.get("annotations", [])}
                self._cache[split] = cached = (mtime, points)
        return cached[1]

    def images(self, split: str) -> dict:
        folder = self._folder(split)
        points = self._points(split)
        items = [
            {"name": p.stem, "file": p.name, "gt": len(points.get(p.stem, [])), "annotated": p.stem in points}
            for p in sorted(folder.iterdir())
            if _is_image(p)
        ]
        return {"split": split, "dir": str(folder), "images": items}

    def points(self, split: str, name: str) -> list:
        return self._points(split).get(name, [])

    def _folder(self, split: str) -> Path:
        _check(split)
        folder = self.data_root / "images" / split
        if not folder.is_dir():
            raise FileNotFoundError(f"no image folder {folder}")
        return folder


def _check(split: str) -> None:
    if not split or "/" in split or "\\" in split or split.startswith("."):
        raise ValueError(f"bad split name {split!r}")


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
