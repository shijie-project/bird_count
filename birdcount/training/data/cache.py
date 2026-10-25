"""Decoded, fitted frames on disk, memory-mapped by every DataLoader worker.

Decoding a JPEG and resizing a 1080p image to the model geometry cost
~8 ms per sample — more than everything else a training worker does — and
the result is the same every epoch. So each image is decoded and `fit_frame`d
once, stored as a raw .npy, and read back with mmap: the pages live in the
OS page cache, shared by all workers instead of copied into each.

The cache key covers the source path, size and mtime and the target
geometry, so editing or replacing an image rebuilds its entry. Files are
written atomically (temp + rename), so concurrent builders cannot leave a
torn entry. The cache is disposable: delete the folder at any time.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from birdcount.core.frames import fit_frame, read_image


logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"-(\d+)x(\d+)\.npy$")


@dataclass(frozen=True)
class CachedFrame:
    frame: np.ndarray  # (H, W, 3) uint8 BGR at the model geometry, read-only memmap
    source_size: tuple[int, int]  # (width, height) of the original image


class FrameCache:
    def __init__(self, root: Path, size: tuple[int, int]):
        self.size = size
        self.dir = root / f"frames_{size[0]}x{size[1]}"
        # Per-process memmap handles; never pickled into DataLoader workers.
        self._open: dict[Path, CachedFrame] = {}

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_open": {}}

    def _key(self, path: Path) -> str:
        st = path.stat()
        ident = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{self.size}"
        return f"{path.stem}-{hashlib.sha1(ident.encode()).hexdigest()[:16]}"

    def get(self, path: Path) -> CachedFrame:
        cached = self._open.get(path)
        if cached is not None:
            return cached
        key = self._key(path)
        hits = sorted(self.dir.glob(f"{glob.escape(key)}-*.npy"))
        entry = hits[0] if hits else self._build(path, key)
        match = _NAME_RE.search(entry.name)
        cached = CachedFrame(
            frame=np.load(entry, mmap_mode="r"), source_size=(int(match.group(1)), int(match.group(2)))
        )
        self._open[path] = cached
        return cached

    def _build(self, path: Path, key: str) -> Path:
        image = read_image(path)
        source_size = (image.shape[1], image.shape[0])
        frame = np.ascontiguousarray(fit_frame(image, self.size))
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = self.dir / f"{key}-{source_size[0]}x{source_size[1]}.npy"
        # The temp name must not match "*.npy", or a concurrent reader could
        # pick up a half-written entry.
        tmp = entry.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            np.save(f, frame)
        os.replace(tmp, entry)
        return entry

    def warm(self, paths: Iterable[Path]) -> None:
        """Build every missing entry now (in the main process, before workers start)."""
        paths = list(paths)
        missing = [p for p in paths if not any(self.dir.glob(f"{glob.escape(self._key(p))}-*.npy"))]
        if missing:
            logger.info("Caching %d of %d frames at %dx%d in %s ...", len(missing), len(paths), *self.size, self.dir)
            for p in missing:
                self._build(p, self._key(p))
