"""A snapshot as the rater sees it: the area outside the camera's mask shaded.

The count only covers the mask's area, so the crowding the model judges must
be the same area; the rest stays visible but dim, for context.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from birdcount.core.camera_identity import index_by_mac
from birdcount.core.masks import read_keep_mask
from birdcount.core.paths import IMAGE_SUFFIXES


SHADE = 0.3  # brightness kept outside the mask
JPEG_QUALITY = 85


class MaskShader:
    def __init__(self, mask_dir: Path | None):
        self.paths: dict[str, Path] = {}
        if mask_dir is not None and mask_dir.is_dir():
            self.paths, _ = index_by_mac(p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self._keep: dict[tuple[str, int, int], np.ndarray | None] = {}

    def keep(self, mac: str, size: tuple[int, int]) -> np.ndarray | None:
        key = (mac, *size)
        if key not in self._keep:
            path = self.paths.get(mac)
            self._keep[key] = read_keep_mask(path, size) if path else None
        return self._keep[key]

    def jpeg(self, file: Path, mac: str) -> bytes:
        img = cv2.imread(str(file))
        if img is None:
            raise ValueError(f"cannot read {file}")
        keep = self.keep(mac, (img.shape[1], img.shape[0]))
        if keep is not None:
            out = img.astype(np.float32)
            out[keep == 0] *= SHADE
            img = out.astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise ValueError(f"cannot encode {file}")
        return buf.tobytes()
