"""Per-camera region masks, loaded once at inferencer startup.

Each camera sees a fixed slice of the shed that is worth counting; the rest
(walkways, feeders, the neighbouring pen, sky) is masked out. The 16 mask PNGs
are hand-drawn per camera and named by bare MAC — see `runtime.camera_identity`.

Why this is not optional cosmetics: the pile-up thresholds in
`configs/alarm.json` were calibrated by the delivery package on **hard-black
masked** input, and the masks drop between 29% and 67% of the frame depending
on the camera. Counting the full frame against those thresholds is a
systematic over-count.

The masks are applied in the same two places as the delivery package's
`model_runtime/inference.py`, and for two different reasons:

    frame     (before normalization)   the model was trained on images with
                                       this region literally black, so feeding
                                       it black is what reproduces training
    density   (before the sum)         clamps whatever the receptive field
                                       still bleeds across the boundary, and
                                       makes the density written to SHM — and
                                       so the monitor heatmap — honest

The density weights come from an INTER_AREA downsample of the keep-mask, so a
density cell straddling the boundary is weighted by the fraction of it that is
inside, rather than being rounded wholly in or out.

Everything expensive (PNG decode, resize) happens once in `load_stream_masks`;
the hot path is one gather and one multiply per batch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch

from runtime.camera_identity import index_by_mac


logger = logging.getLogger(__name__)


# A mask pixel counts as "blocked" when every channel is below this. Matches
# `model_runtime/inference.py:93` (`np.all(mask_img < 10, axis=2)`) so a mask
# drawn for the delivery package means the same thing here.
_BLACK_LEVEL = 10

_MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


class StreamMasks:
    """Keep-masks for every stream, resident on the GPU.

    `frame_keep` is indexed by *global* stream id so the hot path can gather
    with the batch's `sids` list directly. Streams without a mask get all-ones
    rows, which makes them a no-op multiply rather than a branch in the loop.
    """

    def __init__(
        self,
        frame_keep: torch.Tensor,
        keep_u8: np.ndarray,
        covered: dict[int, str],
        uncovered: list[int],
    ):
        self.frame_keep = frame_keep  # (S, 1, H, W) fp16, device-resident
        self.covered = covered  # sid -> MAC
        self.uncovered = uncovered  # sids running unmasked

        # Host-side keep-masks, kept only until the density weights are built
        # (which needs the model's output resolution, known after warmup).
        self._keep_u8: np.ndarray | None = keep_u8  # (S, H, W) uint8 0/1
        self._density_keep: torch.Tensor | None = None
        self._density_hw: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def frame_for(self, sids: Sequence[int]) -> torch.Tensor:
        """Keep-mask rows for one batch, shaped to broadcast over (B, C, H, W)."""
        return self.frame_keep[sids]

    def density_for(self, sids: Sequence[int], hw: tuple[int, int]) -> torch.Tensor:
        """Area-weighted keep-mask at density resolution, built on first use."""
        if self._density_keep is None or self._density_hw != hw:
            self._build_density(hw)
        return self._density_keep[sids]

    # ------------------------------------------------------------------
    # Density weights
    # ------------------------------------------------------------------

    def _build_density(self, hw: tuple[int, int]) -> None:
        """Downsample every keep-mask to the model's output resolution.

        INTER_AREA (not NEAREST) is the point: it yields the fraction of each
        density cell that lies inside the keep region, so a bird straddling the
        boundary contributes proportionally instead of all-or-nothing.
        """
        if self._keep_u8 is None:
            raise RuntimeError("density weights already built and host masks released")
        h, w = hw
        stack = np.stack(
            [cv2.resize(keep.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) for keep in self._keep_u8]
        )
        self._density_keep = (
            torch.from_numpy(stack).to(self.frame_keep.device, dtype=torch.float16).unsqueeze(1).contiguous()
        )
        self._density_hw = hw
        self._keep_u8 = None  # frees ~17 MB of host RAM per worker
        logger.info(
            "Density keep-weights built at %dx%d (%.1f KB on device).",
            h,
            w,
            self._density_keep.numel() * 2 / 1024,
        )


def load_stream_masks(
    mask_dir: str | Path,
    sid_to_mac: dict[int, str | None],
    num_streams: int,
    target_hw: tuple[int, int],
    device: torch.device,
    name: str = "Inferencer",
) -> StreamMasks | None:
    """Load one mask per stream. Returns None when masking is off or unusable.

    Every failure mode degrades to "this stream counts the full frame" plus a
    startup warning naming it — never an exception. A missing mask produces
    counts that are too high, which is visible; a crashed runtime is not.
    """
    if not str(mask_dir).strip():
        return None

    root = Path(mask_dir)
    if not root.is_dir():
        logger.warning("[%s] MASK_DIR=%s is not a directory. Running unmasked.", name, root)
        return None

    files = [p for p in sorted(root.iterdir()) if p.suffix.lower() in _MASK_SUFFIXES]
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
    if not by_mac:
        logger.warning("[%s] No mask files with a recognisable MAC in %s. Running unmasked.", name, root)
        return None

    H, W = target_hw
    keep_u8 = np.ones((num_streams, H, W), dtype=np.uint8)
    covered: dict[int, str] = {}
    uncovered: list[int] = []

    for sid in range(num_streams):
        mac = sid_to_mac.get(sid)
        path = by_mac.get(mac) if mac else None
        if path is None:
            uncovered.append(sid)
            continue
        keep = _load_keep_mask(path, W, H, name)
        if keep is None:
            uncovered.append(sid)
            continue
        keep_u8[sid] = keep
        covered[sid] = mac

    if not covered:
        logger.warning(
            "[%s] %d mask file(s) found in %s but none matched a stream MAC. Running unmasked.",
            name,
            len(by_mac),
            root,
        )
        return None

    frame_keep = torch.from_numpy(keep_u8).to(device, dtype=torch.float16).unsqueeze(1).contiguous()

    kept = float(keep_u8[sorted(covered)].mean())
    logger.info(
        "[%s] Region masks loaded: %d/%d stream(s) masked, mean keep %.1f%% (%.1f MB on device).",
        name,
        len(covered),
        num_streams,
        kept * 100.0,
        frame_keep.numel() * 2 / (1024 * 1024),
    )
    if uncovered:
        logger.warning(
            "[%s] No mask for %d stream(s) — they count the FULL frame, which reads high "
            "against thresholds calibrated on masked input: %s",
            name,
            len(uncovered),
            ", ".join(f"sid={s} ({sid_to_mac.get(s) or 'no MAC'})" for s in uncovered),
        )

    return StreamMasks(frame_keep=frame_keep, keep_u8=keep_u8, covered=covered, uncovered=uncovered)


def _load_keep_mask(path: Path, width: int, height: int, name: str) -> np.ndarray | None:
    """Read one mask PNG and return a (H, W) uint8 keep-mask at frame resolution.

    Resized with NEAREST so the mask stays strictly binary; the frames it will
    multiply get the same geometric squash from `stream_grabber._camera`, so
    the two line up even though that squash is not aspect-preserving.
    """
    # np.fromfile + imdecode rather than imread: cv2.imread cannot open a path
    # with non-ASCII characters on Windows.
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error("[%s] Could not read mask %s: %s", name, path, e)
        return None
    if img is None:
        logger.error("[%s] Could not decode mask %s.", name, path)
        return None

    if (img.shape[1], img.shape[0]) != (width, height):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    keep = (~np.all(img < _BLACK_LEVEL, axis=2)).astype(np.uint8)
    if not keep.any():
        logger.error("[%s] Mask %s is fully black — every pixel would be dropped. Ignoring it.", name, path)
        return None
    return keep
