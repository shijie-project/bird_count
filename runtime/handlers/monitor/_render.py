import logging
import math
import time
from typing import Optional

import cv2
import numpy as np

from runtime.inferencer import InferenceResult
from runtime.shared_memory import SharedMemory

from . import _windows as win


try:
    import tkinter as tk

    root = tk.Tk()
    SCREEN_W, SCREEN_H = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080  # fallback


logger = logging.getLogger(__name__)


class _InternalMonitorRenderer:
    """
    Actual rendering logic that runs in the dedicated DisplayProcess.

    Hot-path design:
        * Tiles are resized directly into the canvas slice (no intermediate
          buffer, no second copy).
        * Heatmap composition uses preallocated buffers — zero allocations
          per call on the steady-state path.
        * Per-tick invariants (flash state, density availability, source
          block selection) are computed once per tick(), not once per tile.
    """

    # --- Style Config (Static constants for cache efficiency) ---
    COLOR_GRID = (50, 50, 50)
    COLOR_BG_BAR = (0, 0, 0)
    COLOR_BG_ALERT = (0, 0, 255)
    COLOR_TEXT_NORMAL = (0, 255, 0)
    COLOR_TEXT_ALERT = (255, 255, 255)

    # (bg, text, border, thickness) — picked once per tile in _draw_overlay.
    _STYLE_NORMAL = (COLOR_BG_BAR, COLOR_TEXT_NORMAL, COLOR_GRID, 1)
    _STYLE_ALERT = (COLOR_BG_ALERT, COLOR_TEXT_ALERT, COLOR_BG_ALERT, 3)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1

    # Reserve room for the OS title bar / taskbar so the bottom row of tiles
    # doesn't get clipped off the visible desktop. Expressed as a fraction of
    # the screen so the reserve scales with DPI (taskbar/titlebar grow on HiDPI).
    CHROME_MARGIN_W_RATIO = 0.01  # ~1% — thin window borders only
    CHROME_MARGIN_H_RATIO = 0.08  # ~8% — taskbar (~4%) + title bar (~3%) + buffer

    # Heatmap overlay tuning. Composed at density (model-output) resolution
    # on CPU and upsampled once.
    HEATMAP_ALPHA_BG = 0.5
    HEATMAP_ALPHA_FG = 0.5
    HEATMAP_THRESHOLD = 0.1
    # Absolute density → uint8 scaler. 255 assumes peak density ≈ 1.0. Bump
    # to e.g. 2550 if your model emits Gaussian splats with peaks ~0.1.
    # Tune by printing density.max() on a few representative frames.
    HEATMAP_SCALE = 255.0

    TICK_POLL_INTERVAL = 0.04  # 25 Hz

    def __init__(
        self,
        num_streams: int,
        window_name="PileUp Monitor",
        name="Monitor",
        show_density_map_flag=None,
    ):
        self.window_name = window_name
        self.name = name
        self.num_streams = num_streams
        self.is_window_setup = False
        self._hwnd: Optional[int] = None

        self.ui_update_interval = self.TICK_POLL_INTERVAL

        self._last_ui_update = 0.0

        # mp.Value(bool) shared with MonitorHandler — toggled by the GUI button.
        # None = always off (no GUI). Polled once per tick in tick().
        self._show_density_map_flag = show_density_map_flag

        # Latest pending result per stream — populated by stage(), consumed by
        # tick(). Each entry's buffer_idx is "claimed" (READING state in SHM)
        # until either superseded by a newer stage() call (immediate ack) or
        # rendered on the next tick (deferred ack). This is what makes the
        # renderer coalesce: inference at 60 FPS only pays for 25 renders/sec.
        self._pending: dict[int, tuple[InferenceResult, int]] = {}

        # Cached uint8 threshold for the heatmap mask (avoids per-call multiply).
        self._heatmap_threshold_u8 = int(self.HEATMAP_THRESHOLD * 255)

        # Heatmap working buffers, keyed by target tile shape (h, w). Grid mode
        # uses tile-sized buffers; zoom mode uses canvas-sized ones. Each shape
        # is allocated once on first use, then reused for free.
        self._heatmap_bufs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Compose buffers used by the same-shape fast path: blend the heatmap
        # onto a copy of the preview at its native resolution, then a single
        # resize promotes the composite to tile / canvas size. Keyed by the
        # full source shape (h, w, 3).
        self._compose_bufs: dict[tuple[int, ...], np.ndarray] = {}

        # Click-to-zoom: when set, the focused stream fills the whole canvas;
        # click again to return to grid view. Mutated only on the OpenCV UI
        # thread (the mouse callback fires from cv2.waitKey, same thread as tick).
        self._focused_sid: Optional[int] = None
        self._layout_dirty = False

        self._init_canvas()

        # Pre-formatted label prefixes — string formatting is cheap but
        # constant work in the inner loop adds up across many streams.
        self._label_prefixes = [f"CAM {i}: " for i in range(self.num_streams)]

    def _init_canvas(self):
        # Layout: square-ish grid, fit inside the visible desktop area.
        usable_w = max(320, SCREEN_W - int(SCREEN_W * self.CHROME_MARGIN_W_RATIO))
        usable_h = max(240, SCREEN_H - int(SCREEN_H * self.CHROME_MARGIN_H_RATIO))
        self.cols = int(math.ceil(math.sqrt(self.num_streams)))
        self.rows = int(math.ceil(self.num_streams / self.cols))
        self.tile_w = usable_w // self.cols
        self.tile_h = usable_h // self.rows

        # Initial window dimensions (the user can still drag-resize at runtime).
        self.window_w = self.cols * self.tile_w
        self.window_h = self.rows * self.tile_h

        self.canvas = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), dtype=np.uint8)

        # For each stream we precompute:
        #   - tile_view: a non-copying slice of the canvas we can resize INTO
        #   - coords: (x1, y1, x2, y2) for overlay drawing
        # Holding the view directly skips a slice op every tick.
        self._tile_views: list[np.ndarray] = []
        self._tile_coords: list[tuple[int, int, int, int]] = []
        for i in range(self.num_streams):
            r, c = divmod(i, self.cols)
            y1, y2 = r * self.tile_h, (r + 1) * self.tile_h
            x1, x2 = c * self.tile_w, (c + 1) * self.tile_w
            self._tile_views.append(self.canvas[y1:y2, x1:x2])
            self._tile_coords.append((x1, y1, x2, y2))

    def _setup_window(self) -> None:
        """Create the OpenCV window and pop it to front once.

        Called lazily on the first tick() that has frames to show, so the
        window doesn't appear before any data is ready. Caches the native
        HWND so later Win32 ops don't need to look it up by title.
        """
        # WINDOW_NORMAL = user can drag-resize; WINDOW_KEEPRATIO preserves
        # aspect ratio so tiles don't stretch when the user resizes.
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, self.window_w, self.window_h)
        cv2.moveWindow(self.window_name, 0, 0)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        # OpenCV creates windows lazily — pump the message loop once so the
        # native HWND exists before we try to look it up.
        cv2.waitKey(1)

        self._hwnd = win.find_hwnd(self.window_name)
        win.disable_close_button(self._hwnd, tag=self.name)
        # Pop to the front once on first show so the user notices the
        # monitor turned on; afterwards it behaves like a normal window
        # and can be hidden behind other apps.
        win.force_foreground(self._hwnd, tag=self.name)

        self.is_window_setup = True

    def stage(self, result: InferenceResult) -> Optional[tuple[int, int]]:
        """Register a new result for rendering on the next tick.

        Returns a (sid, buffer_idx) pair that must be ack'd RIGHT NOW — either
        because the result is out-of-range, or because it supersedes an older
        pending result for the same stream that will never be rendered.
        Returns None when the new buffer is staged for deferred render+ack.
        """
        sid = result.stream_id
        if sid >= self.num_streams:
            # Out-of-range stream — we won't render it, release immediately.
            return (sid, result.buffer_idx)

        prev = self._pending.get(sid)
        self._pending[sid] = (result, result.buffer_idx)
        if prev is None:
            return None
        # Older frame for the same stream is now stale — ack it so SHM frees up.
        return (sid, prev[1])

    def tick(self, shm_client: SharedMemory) -> Optional[list[tuple[int, int]]]:
        """Render the staged frames if the display interval has elapsed.

        Returns the list of (sid, buffer_idx) pairs that can now be ack'd
        (because they've been read out of SHM and copied into the canvas).
        Returns None when it's not yet time to redraw — in that case the
        pending entries stay claimed for the next tick.
        """
        now = time.monotonic()
        if now - self._last_ui_update < self.ui_update_interval:
            return None
        if not self._pending:
            self._last_ui_update = now
            return None

        # --- Per-tick invariants (hoisted out of the per-tile loop) ---
        # Source the tile from the downscaled preview block when available
        # (already ~1/16 the size of the full frame); fall back to the full
        # frame block if a deployment opted out of the preview channel.
        source = shm_client.preview if shm_client.preview is not None else shm_client.frames
        show_density_map = (
            bool(self._show_density_map_flag.value) if self._show_density_map_flag is not None else False
        )
        density_block = shm_client.density if (show_density_map and shm_client.density is not None) else None
        # Flashing is global to the frame — every alerting tile flashes
        # together rather than per-tile time samples being slightly off.
        flash_on = (int(now * 4) & 1) == 0

        # Wipe the canvas on grid<->zoom transitions so leftover pixels from
        # the previous layout don't bleed through the new one.
        if self._layout_dirty:
            self.canvas[:] = 0
            self._layout_dirty = False

        acks: list[tuple[int, int]] = []
        focused = self._focused_sid

        if focused is not None:
            # Zoom mode: focused stream fills the whole canvas. Non-focused
            # pending frames are ack'd but not drawn — they'd be invisible
            # anyway, and holding SHM for them would block the inference path.
            focused_entry = self._pending.pop(focused, None)
            for sid, (_, b_idx) in self._pending.items():
                acks.append((sid, b_idx))
            self._pending.clear()

            if focused_entry is not None:
                result, buffer_idx = focused_entry
                canvas_h, canvas_w = self.canvas.shape[:2]
                density_tile = density_block[focused, buffer_idx] if density_block is not None else None
                self._render_into(source[focused, buffer_idx], density_tile, self.canvas, (canvas_w, canvas_h))
                self._draw_overlay(result, (0, 0, canvas_w, canvas_h), flash_on)
                acks.append((focused, buffer_idx))
        else:
            tile_size = (self.tile_w, self.tile_h)
            canvas_views = self._tile_views
            canvas_coords = self._tile_coords
            for sid, (result, buffer_idx) in self._pending.items():
                density_tile = density_block[sid, buffer_idx] if density_block is not None else None
                self._render_into(source[sid, buffer_idx], density_tile, canvas_views[sid], tile_size)
                self._draw_overlay(result, canvas_coords[sid], flash_on)
                acks.append((sid, buffer_idx))
            self._pending.clear()

        if not self.is_window_setup:
            self._setup_window()

        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)
        self._last_ui_update = now
        return acks

    # ------------------------------------------------------------------
    # Click-to-zoom
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Left-click toggles between grid and zoomed-on-clicked-tile view.

        Runs on the OpenCV UI thread (same thread as tick), so no locking
        is needed against the renderer's state.
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._focused_sid is not None:
            self._focused_sid = None
            self._layout_dirty = True
            return
        # Grid → zoom: map click position to a tile. OpenCV reports (x, y) in
        # image (canvas) coordinates for WINDOW_NORMAL windows.
        if 0 <= x < self.cols * self.tile_w and 0 <= y < self.rows * self.tile_h:
            col = x // self.tile_w
            row = y // self.tile_h
            sid = row * self.cols + col
            if sid < self.num_streams:
                self._focused_sid = sid
                self._layout_dirty = True

    def _render_into(
        self,
        src: np.ndarray,
        density_tile: Optional[np.ndarray],
        dst: np.ndarray,
        dst_size_wh: tuple[int, int],
    ) -> None:
        """Render `src` (+ optional density overlay) into `dst` at `dst_size_wh`.

        Fast path — `src.shape == density_tile.shape`: composite the heatmap
        at the small native resolution, then a single INTER_LINEAR resize lifts
        the composite to `dst`. This collapses the old 3-resize path (source,
        heat, mask) into 1.

        Fallback — shapes differ (or no density): upscale `src` straight to `dst`
        with INTER_LINEAR, then overlay heat at `dst`'s shape if requested.
        """
        if density_tile is not None and density_tile.shape[:2] == src.shape[:2]:
            compose_buf = self._compose_bufs.get(src.shape)
            if compose_buf is None:
                compose_buf = np.empty(src.shape, dtype=np.uint8)
                self._compose_bufs[src.shape] = compose_buf
            np.copyto(compose_buf, src)
            self._compose_heatmap(compose_buf, density_tile)
            cv2.resize(compose_buf, dst_size_wh, dst=dst, interpolation=cv2.INTER_LINEAR)
            return

        cv2.resize(src, dst_size_wh, dst=dst, interpolation=cv2.INTER_LINEAR)
        if density_tile is not None:
            self._compose_heatmap(dst, density_tile)

    def _compose_heatmap(self, tile_bgr: np.ndarray, density: np.ndarray) -> None:
        """Compose density heatmap onto `tile_bgr` in place.

        Density is normalized + thresholded at its native resolution (cheap),
        then upsampled to `tile_bgr`'s shape — unless the shapes already match,
        in which case the upsample is skipped entirely. All working buffers are
        preallocated and cached by target shape.
        """
        # Density is stored as float16 in SHM but OpenCV's NORM_MINMAX → CV_8U
        # path doesn't dispatch for CV_16F; promote to float32 first.
        # Absolute mapping — density values are scaled by a fixed factor so
        # the heatmap reflects real density magnitude, not per-tile relative
        # intensity. Clip handles strays above 1/HEATMAP_SCALE; np.clip
        # accepts float16 natively, so no dtype promotion is needed.
        norm_u8 = np.clip(density * self.HEATMAP_SCALE, 0, 255).astype(np.uint8)

        # Threshold at low resolution; bail out if nothing crosses it.
        mask_lo = (norm_u8 > self._heatmap_threshold_u8).view(np.uint8)
        if not mask_lo.any():
            return

        heat_lo = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

        # Look up (or lazily allocate) the blended-output buffer for this shape.
        h, w = tile_bgr.shape[:2]
        bufs = self._heatmap_bufs.get((h, w))
        if bufs is None:
            bufs = (
                np.empty((h, w, 3), dtype=np.uint8),
                np.empty((h, w), dtype=np.uint8),
                np.empty((h, w, 3), dtype=np.uint8),
            )
            self._heatmap_bufs[(h, w)] = bufs
        heat, mask, blended = bufs

        dh, dw = norm_u8.shape[:2]
        if (dh, dw) == (h, w):
            # Same-shape fast path: no upsample needed, blend directly.
            heat, mask = heat_lo, mask_lo
        else:
            # NEAREST upsample for both — the tile carries text + borders on top
            # so heatmap edge smoothness is invisible. NEAREST is ~2-3× faster
            # than LINEAR on the heat path.
            cv2.resize(heat_lo, (w, h), dst=heat, interpolation=cv2.INTER_NEAREST)
            cv2.resize(mask_lo, (w, h), dst=mask, interpolation=cv2.INTER_NEAREST)

        # Blend whole tile, then mask-copy only the heatmap pixels back.
        # cv2.copyTo with a uint8 mask is a C-loop, much faster than numpy
        # boolean indexing.
        cv2.addWeighted(tile_bgr, self.HEATMAP_ALPHA_BG, heat, self.HEATMAP_ALPHA_FG, 0, dst=blended)
        cv2.copyTo(blended, mask, tile_bgr)

    def _draw_overlay(self, res: InferenceResult, coords: tuple, flash_on: bool) -> None:
        x1, y1, x2, y2 = coords

        # Pick a precomputed style tuple — no per-call tuple construction.
        bg_color, text_color, border_color, thick = (
            self._STYLE_ALERT if (res.alert_flag and flash_on) else self._STYLE_NORMAL
        )

        # Header bar
        cv2.rectangle(self.canvas, (x1, y1), (x1 + 120, y1 + 30), bg_color, -1)

        # Label text — only the count changes; the "CAM N: " prefix is
        # precomputed at __init__.
        label = self._label_prefixes[res.stream_id] + str(int(res.count))
        cv2.putText(
            self.canvas,
            label,
            (x1 + 10, y1 + 15),
            self.FONT,
            self.FONT_SCALE,
            text_color,
            self.FONT_THICKNESS,
        )

        # Tile Border
        cv2.rectangle(self.canvas, (x1, y1), (x2, y2), border_color, thick)
