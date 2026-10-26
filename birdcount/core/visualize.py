"""Drawing density maps and region masks over frames.

One convention everywhere a heatmap is shown (live monitor, video tools,
evaluation overlays): colour is tied to the density value itself, on a fixed
scale, so a tile's colour is comparable to other tiles' and to the same
camera a minute ago. Never normalise per frame.

numpy + cv2 only: no torch.
"""

from __future__ import annotations

import cv2
import numpy as np


# Density (birds per output cell) mapped to the top of the colour map; above
# it clips to red. Measured on the val split: cells are 0 almost everywhere,
# the 99th percentile is ~0.07, and per-frame peaks land in 0.12-0.22 — one
# bird's mass spreads over ~6-7 cells, so even an isolated bird peaks well
# below 1.0. At 0.12 bird cores run warm, crowds merge into broad red areas,
# and a real pile-up is what pushes cells past it. Re-measure after
# retraining or changing the output stride (the unit is per cell).
HEATMAP_VMAX = 0.12
# Cells at or below this stay uncoloured instead of washing the frame blue.
HEATMAP_MIN_DENSITY = 0.02
HEATMAP_ALPHA = 0.5

# Region-mask overlay: the ignored area is tinted and the keep boundary
# outlined, so a mask off by a few pixels is still obvious at thumbnail size.
MASK_TINT = (60, 0, 160)  # BGR dark magenta, distinct from the JET heatmap
MASK_TINT_ALPHA = 0.55
MASK_EDGE_COLOR = (0, 255, 255)  # BGR yellow
_EDGE_KERNEL = np.ones((3, 3), np.uint8)

TEXT_COLOR = (255, 255, 255)


class Scratch:
    """Reusable work buffers for drawing overlays every frame (the live monitor).

    Without one, every call allocates its tile-sized intermediates afresh:
    fine for a figure, ~40 % of the cost when 20 tiles are redrawn 25 times a
    second. One per drawing thread.
    """

    def __init__(self):
        self._bufs: dict[tuple, np.ndarray] = {}

    def get(self, name: str, shape: tuple[int, ...], fill=None) -> np.ndarray:
        """The buffer `name` at `shape`; `fill` is written once, when it is created."""
        key = (name, shape)
        buf = self._bufs.get(key)
        if buf is None:
            buf = self._bufs[key] = np.empty(shape, np.uint8)
            if fill is not None:
                buf[:] = fill
        return buf


def _buf(scratch: Scratch | None, name: str, shape: tuple[int, ...], fill=None) -> np.ndarray:
    if scratch is not None:
        return scratch.get(name, shape, fill)
    buf = np.empty(shape, np.uint8)
    if fill is not None:
        buf[:] = fill
    return buf


def density_to_heatmap(
    density: np.ndarray, *, vmax: float = HEATMAP_VMAX, min_density: float = HEATMAP_MIN_DENSITY
) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) density -> ((H, W, 3) BGR JET heatmap, (H, W) uint8 mask 255 where coloured)."""
    density = np.asarray(density)
    if density.ndim == 3:
        density = density.squeeze()
    if density.ndim != 2:
        raise ValueError(f"density must be 2D, got shape {density.shape}")
    scale = 255.0 / max(float(vmax), 1e-12)
    # Negative -> 0, then scale with a saturating, rounding cast to uint8: two
    # passes instead of numpy's four (4x faster at 720p; this runs per frame).
    positive = cv2.threshold(density.astype(np.float32, copy=False), 0, 0, cv2.THRESH_TOZERO)[1]
    norm = cv2.convertScaleAbs(positive, alpha=scale)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # Threshold in u8 space so the mask matches exactly what was coloured.
    mask = cv2.threshold(norm, int(min_density * scale), 255, cv2.THRESH_BINARY)[1]
    return heat, mask


def overlay_density(
    frame: np.ndarray,
    density: np.ndarray,
    *,
    smooth: bool = True,
    vmax: float = HEATMAP_VMAX,
    alpha: float = HEATMAP_ALPHA,
    scratch: Scratch | None = None,
) -> np.ndarray:
    """Blend the heatmap over `frame` in place (only where density is visible); returns `frame`.

    smooth=True upsamples the density (LINEAR) before colouring — the right
    picture. smooth=False colours at density resolution and upsamples the
    colours (NEAREST): blockier, several times cheaper, for small tiles.
    (Upsampling colours LINEARly would blend between colour-map entries and
    invent colours that are not on the scale.)
    """
    h, w = frame.shape[:2]
    density = np.asarray(density, dtype=np.float32).squeeze()
    if smooth:
        if density.shape != (h, w):
            density = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
        heat, mask = density_to_heatmap(density, vmax=vmax)
        if not mask.any():
            return frame
    else:
        heat_lo, mask_lo = density_to_heatmap(density, vmax=vmax)
        if not mask_lo.any():  # decided at density resolution, before any upsampling
            return frame
        heat, mask = heat_lo, mask_lo
        if heat_lo.shape[:2] != (h, w):
            heat = cv2.resize(heat_lo, (w, h), dst=_buf(scratch, "heat", (h, w, 3)), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask_lo, (w, h), dst=_buf(scratch, "heat_mask", (h, w)), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(frame, 1.0 - alpha, heat, alpha, 0, dst=_buf(scratch, "blend", frame.shape))
    cv2.copyTo(blended, mask, frame)
    return frame


def mask_layers(keep: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(ignored, edge) uint8 masks for `overlay_mask` at `size` = (width, height).

    Separate so a caller drawing the same mask every frame can compute them once.
    """
    width, height = size
    if keep.shape != (height, width):
        keep = cv2.resize(keep, (width, height), interpolation=cv2.INTER_NEAREST)
    ignored = (keep == 0).astype(np.uint8)
    edge = cv2.morphologyEx(keep, cv2.MORPH_GRADIENT, _EDGE_KERNEL)
    return ignored, edge


def overlay_mask(
    frame: np.ndarray, layers: tuple[np.ndarray, np.ndarray], *, scratch: Scratch | None = None
) -> np.ndarray:
    """Tint the ignored region and outline the keep boundary, in place; returns `frame`."""
    ignored, edge = layers
    tint = _buf(scratch, "mask_tint", frame.shape, MASK_TINT)
    blended = cv2.addWeighted(
        frame, 1.0 - MASK_TINT_ALPHA, tint, MASK_TINT_ALPHA, 0, dst=_buf(scratch, "blend", frame.shape)
    )
    cv2.copyTo(blended, ignored, frame)
    cv2.copyTo(_buf(scratch, "mask_edge", frame.shape, MASK_EDGE_COLOR), edge, frame)
    return frame


def draw_text(image: np.ndarray, text: str, org: tuple[int, int], scale: float, color=TEXT_COLOR) -> None:
    """Text with a black outline, legible over any frame or heatmap colour."""
    thick = max(1, round(scale * 2))
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
