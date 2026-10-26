"""Composing the monitor picture: every camera in a grid, or one zoomed in.

Each tile is redrawn only when something it shows changed — a newer frame,
the count, an overlay switched, the alert flash — so a quiet camera costs
nothing per refresh (v2 rescaled all 20 frames on every 25 Hz tick). Frames
are read without claiming their slot; a frame overwritten mid-copy is caught
by its `frame_idx` and redrawn next time. Heatmap and region-mask overlays
come from `core.visualize`, so the monitor reads like the evaluation figures.

numpy + cv2 only.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from birdcount.core.visualize import Scratch, mask_layers, overlay_density, overlay_mask
from birdcount.deploy.shm import FrameRing


_GRID = (50, 50, 50)
_HEADER_BG = (0, 0, 0)
_ALERT = (0, 0, 255)
_TEXT = (0, 255, 0)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class GridRenderer:
    def __init__(
        self,
        num_streams: int,
        canvas_size: tuple[int, int],
        *,
        labels: list[str] | None = None,
        keep_masks: dict[int, np.ndarray] | None = None,
    ):
        """`keep_masks[sid]`: the camera's keep-mask at frame size, for the mask overlay."""
        self.num_streams = num_streams
        self.cols = math.ceil(math.sqrt(num_streams))
        self.rows = math.ceil(num_streams / self.cols)
        width, height = canvas_size
        self.tile_w, self.tile_h = width // self.cols, height // self.rows
        self.canvas = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), np.uint8)
        self.labels = labels or [f"CAM {i}" for i in range(num_streams)]
        self.keep_masks = keep_masks
        self.focused: int | None = None
        self._tiles = []  # views into the canvas: a tile is resized straight into place
        for sid in range(num_streams):
            r, c = divmod(sid, self.cols)
            x1, y1 = c * self.tile_w, r * self.tile_h
            self._tiles.append(self.canvas[y1 : y1 + self.tile_h, x1 : x1 + self.tile_w])
        self._drawn: dict[int, tuple] = {}  # sid (-1: the zoomed view) -> what it last showed
        self._layers: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._scratch = Scratch()
        self._headers: dict[int, tuple[tuple, np.ndarray]] = {}  # sid -> (what it shows, the bar)
        self.redraws = 0  # tiles actually drawn, for tests and diagnostics

    def tile_at(self, x: int, y: int) -> int | None:
        """The camera whose tile covers canvas pixel (x, y)."""
        if not (0 <= x < self.cols * self.tile_w and 0 <= y < self.rows * self.tile_h):
            return None
        sid = (y // self.tile_h) * self.cols + x // self.tile_w
        return sid if sid < self.num_streams else None

    def toggle_zoom(self, x: int, y: int) -> None:
        """Double-click: zoom into the tile under (x, y), or back out to the grid."""
        self.focused = None if self.focused is not None else self.tile_at(x, y)
        self.canvas[:] = 0
        self._drawn.clear()

    def render(self, ring: FrameRing, *, flash_on: bool) -> bool:
        """Bring the canvas up to date; True if any pixel changed."""
        density, mask = ring.show_density, ring.show_mask
        if self.focused is not None:
            return self._draw(ring, self.focused, -1, self.canvas, density, mask, flash_on, smooth=True)
        changed = False
        for sid, view in enumerate(self._tiles):
            changed |= self._draw(ring, sid, sid, view, density, mask, flash_on, smooth=False)
        return changed

    def _draw(self, ring, sid, key_id, dst, show_density, show_mask, flash_on, *, smooth) -> bool:
        latest = ring.latest(sid)
        if latest is None:
            return False
        slot, frame_idx = latest
        status = ring.status[sid]
        count, alert, density_slot = float(status["count"]), bool(status["alert"]), int(status["slot"])
        inferred = int(status["frame_idx"]) if show_density else None  # a new heatmap
        key = (frame_idx, inferred, round(count), show_density, show_mask, alert and flash_on, alert)
        if self._drawn.get(key_id) == key:
            return False

        h, w = dst.shape[:2]
        cv2.resize(ring.frames[sid, slot], (w, h), dst=dst, interpolation=cv2.INTER_LINEAR)
        if int(ring.meta["frame_idx"][sid, slot]) != frame_idx:
            return False  # overwritten while copying: try again next refresh
        if show_density and density_slot >= 0:
            overlay_density(dst, ring.density[sid, density_slot], smooth=smooth, scratch=self._scratch)
        scale = min(max(h / 216, 0.5), 1.25)  # sized for a ~384x216 grid tile; capped so big tiles keep the picture
        if show_mask:
            if self.keep_masks is not None and sid in self.keep_masks:
                overlay_mask(dst, self._mask_layers(sid, w, h), scratch=self._scratch)
            else:
                _text(dst, "NO MASK", (int(15 * scale), h - int(15 * scale)), 0.9 * scale, _ALERT)
        self._decorate(dst, sid, count, alert and flash_on, scale)
        self._drawn[key_id] = key
        self.redraws += 1
        return True

    def _mask_layers(self, sid: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        layers = self._layers.get((sid, w, h))
        if layers is None:
            layers = self._layers[(sid, w, h)] = mask_layers(self.keep_masks[sid], (w, h))
        return layers

    def _decorate(self, dst: np.ndarray, sid: int, count: float, flashing: bool, scale: float) -> None:
        h, w = dst.shape[:2]
        border, thick = (_ALERT, 3) if flashing else (_GRID, 1)
        header = self._header(sid, round(count), flashing, scale, w)
        dst[: header.shape[0], : header.shape[1]] = header
        cv2.rectangle(dst, (0, 0), (w - 1, h - 1), border, thick)

    def _header(self, sid: int, count: int, flashing: bool, scale: float, width: int) -> np.ndarray:
        """The "CAM i: n" bar, rendered once per change and pasted on every redraw:
        anti-aliased text at large-tile scale costs more than the frame resize."""
        key = (count, flashing, scale, width)
        cached = self._headers.get(sid)
        if cached is not None and cached[0] == key:
            return cached[1]
        bg, fg = (_ALERT, _WHITE) if flashing else (_HEADER_BG, _TEXT)
        header = np.empty((int(50 * scale) + 1, min(int(240 * scale) + 1, width), 3), np.uint8)
        header[:] = bg
        _text(header, f"{self.labels[sid]}: {count}", (int(15 * scale), int(35 * scale)), 1.2 * scale, fg)
        self._headers[sid] = (key, header)
        return header


def _text(dst: np.ndarray, text: str, org: tuple[int, int], scale: float, color) -> None:
    cv2.putText(dst, text, org, _FONT, scale, color, max(1, round(2 * scale)), cv2.LINE_AA)
