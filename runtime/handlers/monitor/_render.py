import logging
import math
import time
from typing import Optional

import cv2
import numpy as np

from runtime.inferencer import InferenceResult
from runtime.shared_memory import BufferState, SharedMemory
from utils import density_to_heatmap

from . import _windows as win


logger = logging.getLogger(__name__)


# Fallback used when Tk can't open a display (headless / X11-unavailable).
_FALLBACK_SCREEN_SIZE = (1920, 1080)

_cached_screen_size: Optional[tuple[int, int]] = None


def _get_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary display, probed lazily once.

    Deferred to first call so merely importing this module does not spin up
    a transient Tk root in every process that touches it (most notably the
    DisplayProcess child).
    """
    global _cached_screen_size
    if _cached_screen_size is not None:
        return _cached_screen_size
    try:
        import tkinter as tk

        root = tk.Tk()
        try:
            _cached_screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
        finally:
            root.destroy()
    except Exception:
        _cached_screen_size = _FALLBACK_SCREEN_SIZE
    return _cached_screen_size


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
    FONT_SCALE = 1.2
    FONT_THICKNESS = 2

    # Header bar / text geometry, sized to fit FONT_SCALE above. Bumped up so
    # the "CAM N: count" label stays legible after the window is scaled down.
    HEADER_W = 240
    HEADER_H = 50
    TEXT_ORIGIN = (15, 35)  # (dx, dy) from the tile's top-left corner

    # Reserve room for the OS title bar / taskbar so the bottom row of tiles
    # doesn't get clipped off the visible desktop. Expressed as a fraction of
    # the screen so the reserve scales with DPI (taskbar/titlebar grow on HiDPI).
    CHROME_MARGIN_W_RATIO = 0.01  # ~1% — thin window borders only
    CHROME_MARGIN_H_RATIO = 0.08  # ~8% — taskbar (~4%) + title bar (~3%) + buffer

    # Heatmap overlay-blend weights. The density→color logic (normalization,
    # threshold, colormap) lives in `utils.density_to_heatmap` so test.py,
    # side-by-side viz, and this live monitor share one visualization
    # convention. Tune via the constants in utils.py.
    HEATMAP_ALPHA_BG = 0.5
    HEATMAP_ALPHA_FG = 0.5

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

        # Latest inference result per stream — the count/alert to overlay.
        # Updated by stage() as batches arrive (at inference rate) and read by
        # tick() on every render. The frame *pixels* are pulled straight from
        # SHM in tick() (see _latest_frame_idx), so display is fully decoupled
        # from inference: the video renders at the tick rate no matter how
        # often inference produces a new count.
        self._last_result: dict[int, InferenceResult] = {}

        # Heatmap working buffers, keyed by target tile shape (h, w). Grid mode
        # uses tile-sized buffers; zoom mode uses canvas-sized ones. Each shape
        # is allocated once on first use, then reused for free. Bounded in
        # practice by the small number of distinct shapes a fixed-config
        # deployment touches (typically 2: grid tile size + zoomed canvas
        # size); not safe for unbounded shape diversity.
        self._heatmap_bufs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Compose buffers used by the same-shape fast path: blend the heatmap
        # onto a copy of the preview at its native resolution, then a single
        # resize promotes the composite to tile / canvas size. Keyed by the
        # full source shape (h, w, 3). Same bounded-shapes assumption.
        self._compose_bufs: dict[tuple[int, ...], np.ndarray] = {}

        # Smooth-path (zoom mode) cache for the alpha-blended output buffer,
        # keyed by (h, w). Distinct from `_heatmap_bufs` because the smooth
        # path doesn't need separate cached heat/mask buffers — those come
        # fresh from `density_to_heatmap` on the LINEAR-upsampled density —
        # and zoom mode touches at most one (h, w) at a time.
        self._smooth_compose_bufs: dict[tuple[int, int], np.ndarray] = {}

        # Double-click-to-zoom: when set, the focused stream fills the whole
        # canvas (rendered from the full-resolution frame, not the preview);
        # double-click again to return to grid view. Mutated only on the
        # OpenCV UI thread (the mouse callback fires from cv2.waitKey, same
        # thread as tick).
        self._focused_sid: Optional[int] = None
        self._layout_dirty = False

        self._init_canvas()

        # Pre-formatted label prefixes — string formatting is cheap but
        # constant work in the inner loop adds up across many streams.
        self._label_prefixes = [f"CAM {i}: " for i in range(self.num_streams)]

    def _init_canvas(self):
        # Layout: square-ish grid, fit inside the visible desktop area.
        screen_w, screen_h = _get_screen_size()
        usable_w = max(320, screen_w - int(screen_w * self.CHROME_MARGIN_W_RATIO))
        usable_h = max(240, screen_h - int(screen_h * self.CHROME_MARGIN_H_RATIO))
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

    def stage(self, result: InferenceResult) -> None:
        """Record the latest count/alert for a stream.

        Frames are no longer claimed here — the renderer pulls live frames from
        SHM in tick(). We only keep the most recent inference result per stream
        so tick() can overlay the count/alert. Returns None; the DisplayProcess
        treats a None return as "no SHM buffer to ack".
        """
        if result.stream_id < self.num_streams:
            self._last_result[result.stream_id] = result
        return None

    def tick(self, shm_client: SharedMemory) -> None:
        """Draw the freshest SHM frame for every stream at the display rate.

        Fully decoupled from inference: each stream's newest fully-written slot
        is pulled straight from SHM and drawn, with the last known count/alert
        overlaid on top. Inference still runs at its own FPS-capped rate and
        only refreshes the overlaid numbers — the video itself plays as
        smoothly as the grabber fills SHM (typically the video's native FPS).
        Returns None (the monitor no longer pins or acks SHM buffers).
        """
        now = time.monotonic()
        if now - self._last_ui_update < self.ui_update_interval:
            return None
        self._last_ui_update = now

        frames = shm_client.frames
        if frames is None:
            return None

        # --- Per-tick invariants (hoisted out of the per-tile loop) ---
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

        drew_any = False
        focused = self._focused_sid

        if focused is not None:
            # Zoom mode: focused stream fills the whole canvas, from the
            # full-resolution frame (the tile-scale path is too soft here).
            idx = self._latest_frame_idx(shm_client, focused)
            if idx != -1:
                canvas_h, canvas_w = self.canvas.shape[:2]
                density_tile = density_block[focused, idx] if density_block is not None else None
                self._render_into(frames[focused, idx], density_tile, self.canvas, (canvas_w, canvas_h), smooth=True)
                self._draw_overlay(focused, (0, 0, canvas_w, canvas_h), flash_on)
                drew_any = True
        else:
            tile_size = (self.tile_w, self.tile_h)
            for sid in range(self.num_streams):
                idx = self._latest_frame_idx(shm_client, sid)
                if idx == -1:
                    continue
                density_tile = density_block[sid, idx] if density_block is not None else None
                self._render_into(frames[sid, idx], density_tile, self._tile_views[sid], tile_size, smooth=False)
                self._draw_overlay(sid, self._tile_coords[sid], flash_on)
                drew_any = True

        # Don't pop an empty window before any stream has produced a frame.
        if not drew_any and not self.is_window_setup:
            return None

        if not self.is_window_setup:
            self._setup_window()

        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)
        return None

    def _latest_frame_idx(self, shm_client: SharedMemory, sid: int) -> int:
        """Index of the newest fully-written SHM slot for `sid`, or -1 if none.

        Considers slots in READY or READING state (both hold a complete frame);
        WRITING / FREE slots are skipped so we never read a half-written frame.
        Purely a read — the monitor is a passive viewer and does not touch the
        grabber/inference buffer handshake, so it can never starve inference.
        """
        meta = shm_client.stream_metadata[sid]
        states = meta["state"]
        mask = (states == BufferState.READY) | (states == BufferState.READING)
        if not mask.any():
            return -1
        idxs = np.flatnonzero(mask)
        return int(idxs[np.argmax(meta["frame_idx"][idxs])])

    # ------------------------------------------------------------------
    # Click-to-zoom
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        """Left-double-click toggles between grid and zoomed-on-clicked-tile view.

        Runs on the OpenCV UI thread (same thread as tick), so no locking
        is needed against the renderer's state.
        """
        if event != cv2.EVENT_LBUTTONDBLCLK:
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
        smooth: bool = False,
    ) -> None:
        """Render `src` (+ optional density overlay) into `dst` at `dst_size_wh`.

        Fast path — `src.shape == density_tile.shape`: composite the heatmap
        at the small native resolution, then a single INTER_LINEAR resize lifts
        the composite to `dst`. This collapses the old 3-resize path (source,
        heat, mask) into 1.

        Fallback — shapes differ (or no density): upscale `src` straight to `dst`
        with INTER_LINEAR, then overlay heat at `dst`'s shape if requested.

        `smooth` is forwarded to `_compose_heatmap` and selects the
        quality-first upsampling path (LINEAR-upsample raw density before
        colorization). See that method's docstring for the tradeoff.
        """
        if density_tile is not None and density_tile.shape[:2] == src.shape[:2]:
            compose_buf = self._compose_bufs.get(src.shape)
            if compose_buf is None:
                compose_buf = np.empty(src.shape, dtype=np.uint8)
                self._compose_bufs[src.shape] = compose_buf
            np.copyto(compose_buf, src)
            self._compose_heatmap(compose_buf, density_tile, smooth=smooth)
            cv2.resize(compose_buf, dst_size_wh, dst=dst, interpolation=cv2.INTER_LINEAR)
            return

        cv2.resize(src, dst_size_wh, dst=dst, interpolation=cv2.INTER_LINEAR)
        if density_tile is not None:
            self._compose_heatmap(dst, density_tile, smooth=smooth)

    def _compose_heatmap(self, tile_bgr: np.ndarray, density: np.ndarray, smooth: bool = False) -> None:
        """Compose density heatmap onto `tile_bgr` in place.

        Two upsampling strategies, switched by `smooth`:

        Fast path (`smooth=False`, grid mode) — colorize density at its
            native (low) resolution, then NEAREST-upsample heat + mask into
            preallocated tile-sized buffers and blend. NEAREST is ~2-3× faster
            than LINEAR on the heat path; the blockiness it produces is
            largely masked by the tile's text + border at grid scale. We
            cannot safely use LINEAR here because linearly interpolating an
            already-colorized BGR heatmap blends between JET LUT entries
            (e.g., red ↔ blue) and produces colors that don't exist on the
            colormap — visually wrong even though smooth.

        Smooth path (`smooth=True`, zoom mode) — LINEAR-upsample the raw
            scalar density to (h, w) first, then run `density_to_heatmap` on
            the upsampled density, then blend. Interpolating the scalar field
            before colorization avoids the LUT-blending artifact above, at
            the cost of running `applyColorMap` over tile-resolution pixels
            (~20× more pixels than the fast path). Only worth it when one
            stream fills the full canvas — at grid tile scale the extra
            quality is invisible under the overlay text/borders.

        Bails out cheaply if no density cell crosses the heatmap threshold.
        """
        if smooth:
            h, w = tile_bgr.shape[:2]
            # cv2.resize on float16 is unsupported on some OpenCV builds; cast
            # to float32 once — density blocks are small (~kB) so the alloc is
            # cheap and only runs in zoom mode.
            density_f32 = density.astype(np.float32, copy=False)
            if density_f32.shape[:2] != (h, w):
                density_up = cv2.resize(density_f32, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                density_up = density_f32
            heat, mask = density_to_heatmap(density_up)
            if not mask.any():
                return
            blended = self._smooth_compose_bufs.get((h, w))
            if blended is None:
                blended = np.empty((h, w, 3), dtype=np.uint8)
                self._smooth_compose_bufs[(h, w)] = blended
            cv2.addWeighted(tile_bgr, self.HEATMAP_ALPHA_BG, heat, self.HEATMAP_ALPHA_FG, 0, dst=blended)
            cv2.copyTo(blended, mask, tile_bgr)
            return

        heat_lo, mask_lo = density_to_heatmap(density)

        # Cheap early-out: if no cell crosses the threshold, nothing to draw.
        if not mask_lo.any():
            return

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

        dh, dw = heat_lo.shape[:2]
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

    def _draw_overlay(self, sid: int, coords: tuple, flash_on: bool) -> None:
        x1, y1, x2, y2 = coords

        # Count/alert come from the most recent inference result for this
        # stream (may lag the displayed frame by a few frames — that's the
        # point of decoupling). No result yet → show 0 and no alert.
        res = self._last_result.get(sid)
        count = int(res.count) if res is not None else 0
        alert = bool(res.alert_flag) if res is not None else False

        # Pick a precomputed style tuple — no per-call tuple construction.
        bg_color, text_color, border_color, thick = self._STYLE_ALERT if (alert and flash_on) else self._STYLE_NORMAL

        # Header bar
        cv2.rectangle(self.canvas, (x1, y1), (x1 + self.HEADER_W, y1 + self.HEADER_H), bg_color, -1)

        # Label text — only the count changes; the "CAM N: " prefix is
        # precomputed at __init__.
        label = self._label_prefixes[sid] + str(count)
        cv2.putText(
            self.canvas,
            label,
            (x1 + self.TEXT_ORIGIN[0], y1 + self.TEXT_ORIGIN[1]),
            self.FONT,
            self.FONT_SCALE,
            text_color,
            self.FONT_THICKNESS,
        )

        # Tile Border
        cv2.rectangle(self.canvas, (x1, y1), (x2, y2), border_color, thick)
