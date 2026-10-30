"""Per-camera region masks: which part of each frame is counted.

Each camera sees a fixed slice of the shed worth counting; the rest
(walkways, feeders, the neighbouring pen, sky) is masked out by a hand-drawn
PNG named by the camera's bare MAC (see `camera_identity`). White = count,
black = ignore.

Not cosmetic: the pile-up thresholds were calibrated on hard-black masked
input, and the masks drop 29-67% of the frame, so counting the full frame is
a systematic over-count. `core.inference` applies a mask twice — to the frame
(the model was trained on images with this region literally black) and to the
density map before the sum (clamps what the receptive field bleeds across the
boundary). The density weights are an INTER_AREA downsample of the keep-mask,
so a cell straddling the boundary counts by the fraction of it inside.

Loading by MAC degrades rather than fails: a stream with no usable mask counts
the full frame and is named in a warning (too-high counts are visible; a
runtime that refuses to start is not). A mask file asked for explicitly by
path, on the other hand, must load.

numpy + cv2 only: no torch.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from birdcount.core.camera_identity import extract_mac, index_by_mac
from birdcount.core.frames import read_image
from birdcount.core.paths import IMAGE_SUFFIXES


logger = logging.getLogger(__name__)

# A mask pixel is "ignore" when every channel is below this — the delivery
# package's rule (`np.all(mask < 10, axis=2)`), so its masks mean the same here.
BLACK_LEVEL = 10


def read_keep_mask(path: str | os.PathLike[str], size: tuple[int, int]) -> np.ndarray:
    """(H, W) uint8 keep-mask (1 = count) at frame `size` = (width, height).

    Resized with NEAREST so it stays binary; frames get the same
    non-aspect-preserving squash in `frames.fit_frame`, so the two line up.
    Raises if the file is unreadable or the mask keeps nothing.
    """
    img = read_image(path)
    width, height = size
    if (img.shape[1], img.shape[0]) != (width, height):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    keep = (~np.all(img < BLACK_LEVEL, axis=2)).astype(np.uint8)
    if not keep.any():
        raise ValueError(f"mask {Path(path).name} is fully black: it would drop every pixel")
    return keep


def density_weights(keep: np.ndarray, stride: int) -> np.ndarray:
    """Fraction of each stride x stride density cell inside the keep region.

    `keep` is (H, W) or (S, H, W) uint8; the result is float32 at
    (H/stride, W/stride), same leading dims.
    """
    if keep.ndim == 2:
        return density_weights(keep[None], stride)[0]
    h, w = keep.shape[1] // stride, keep.shape[2] // stride
    return np.stack([cv2.resize(k.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) for k in keep])


@dataclass(frozen=True)
class RegionMasks:
    """One keep-mask row per stream; rows without a mask are all ones (a no-op).

    `macs[i]` is the camera whose mask filled row i, None for an all-ones row.
    """

    keep: np.ndarray  # (S, H, W) uint8, 1 = count
    macs: tuple[str | None, ...] = field(default=())

    @property
    def covered(self) -> dict[int, str]:
        return {i: mac for i, mac in enumerate(self.macs) if mac is not None}

    @property
    def uncovered(self) -> list[int]:
        return [i for i, mac in enumerate(self.macs) if mac is None]

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], size: tuple[int, int]) -> RegionMasks:
        """A single mask named explicitly by path. Raises if it cannot be used."""
        keep = read_keep_mask(path, size)
        return cls(keep=keep[None], macs=(extract_mac(Path(path).stem) or Path(path).stem,))

    @classmethod
    def from_dir(
        cls,
        mask_dir: str | os.PathLike[str] | None,
        macs: Sequence[str | None],
        size: tuple[int, int],
        *,
        name: str = "masks",
    ) -> RegionMasks | None:
        """Row i holds the mask in `mask_dir` whose filename carries `macs[i]`.

        None when masking is off (no dir), unusable, or no stream matched —
        every case logged.
        """
        if mask_dir is None or not str(mask_dir).strip():
            return None
        root = Path(mask_dir)
        if not root.is_dir():
            logger.warning("[%s] MASK_DIR=%s is not a directory. Running unmasked.", name, root)
            return None

        files = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        by_mac, ambiguous = index_by_mac(files)
        for mac, matches in ambiguous.items():
            logger.warning(
                "[%s] MAC %s is claimed by %d mask files (%s); ignoring all of them. Rename the one you want to %s.png.",
                name,
                mac,
                len(matches),
                ", ".join(p.name for p in matches),
                mac,
            )

        width, height = size
        keep = np.ones((len(macs), height, width), dtype=np.uint8)
        row_macs: list[str | None] = []
        for i, mac in enumerate(macs):
            path = by_mac.get(mac) if mac else None
            if path is not None:
                try:
                    keep[i] = read_keep_mask(path, size)
                except (OSError, ValueError) as e:
                    logger.error("[%s] Ignoring mask %s: %s", name, path.name, e)
                    path = None
            row_macs.append(mac if path is not None else None)

        masks = cls(keep=keep, macs=tuple(row_macs))
        if not masks.covered:
            logger.warning(
                "[%s] %d mask file(s) in %s but none matched a stream's MAC. Running unmasked.", name, len(files), root
            )
            return None
        logger.info(
            "[%s] Region masks: %d/%d stream(s) masked, mean keep %.1f%%.",
            name,
            len(masks.covered),
            len(macs),
            float(keep[sorted(masks.covered)].mean()) * 100.0,
        )
        if masks.uncovered:
            logger.warning(
                "[%s] No mask for %d stream(s) — they count the FULL frame, which reads high against "
                "thresholds calibrated on masked input: %s",
                name,
                len(masks.uncovered),
                ", ".join(f"#{i} ({macs[i] or 'no MAC'})" for i in masks.uncovered),
            )
        return masks

    @classmethod
    def for_source(
        cls, mask_dir: str | os.PathLike[str] | None, source: str | os.PathLike[str], size: tuple[int, int]
    ) -> RegionMasks | None:
        """The mask for one video / image, found by the MAC in its filename."""
        mac = extract_mac(Path(source).name)
        if mac is None:
            logger.warning("No camera MAC in %s; running unmasked.", Path(source).name)
            return None
        return cls.from_dir(mask_dir, [mac], size, name=Path(source).stem)


def mask_digests(mask_dir: str | os.PathLike[str] | None) -> dict[str, str]:
    """{MAC: sha256 prefix} of the masks in `mask_dir`: what a count was made with.

    Counts change with the mask, so whatever records counts (the count log,
    a replay) names the masks behind them.
    """
    if mask_dir is None or not Path(mask_dir).is_dir():
        return {}
    paths, _ = index_by_mac(p for p in Path(mask_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return {mac: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for mac, p in sorted(paths.items())}
