"""The monitor process and its consumer-side handler."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from birdcount.core.masks import RegionMasks
from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.base import Batch, Handler
from birdcount.deploy.lifecycle import init_child
from birdcount.deploy.monitor import windows
from birdcount.deploy.monitor.render import GridRenderer
from birdcount.deploy.shm import FrameRing, RingLayout


logger = logging.getLogger(__name__)

WINDOW = "PileUp Monitor"
_CHROME = (0.01, 0.08)  # share of the screen left for borders, title bar and taskbar
_TERMINATE_GRACE = 2.0


class MonitorProcess(mp.Process):
    def __init__(
        self,
        layout: RingLayout,
        *,
        fps: float,
        labels: Sequence[str],
        mask_dir: Path | None,
        macs: Sequence[str | None],
        log_level: str = "INFO",
        log_file: Path | None = None,
    ):
        super().__init__(name="Monitor", daemon=True)
        self.layout = layout
        self.fps = fps
        self.labels = list(labels)
        self.mask_dir = mask_dir
        self.macs = list(macs)
        self.log_level = log_level
        self.log_file = log_file
        self.stop_event = mp.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        init_child(self.log_level, self.log_file, self.stop_event)
        ring = FrameRing.attach(self.layout)
        sw, sh = windows.screen_size()
        size = (int(sw * (1 - _CHROME[0])), int(sh * (1 - _CHROME[1])))
        renderer = GridRenderer(self.layout.num_streams, size, labels=self.labels)
        ring.display_fps = self.fps  # the grabber now retrieves frames at this rate too
        shown = False
        try:
            period = 1.0 / self.fps
            next_frame = time.monotonic()
            while not self.stop_event.is_set():
                if ring.show_mask and renderer.keep_masks is None:
                    renderer.keep_masks = self._keep_masks()
                changed = renderer.render(ring, flash_on=int(time.monotonic() * 4) % 2 == 0)
                if changed and not shown:
                    shown = self._open_window(renderer)
                if changed:
                    cv2.imshow(WINDOW, renderer.canvas)
                cv2.waitKey(1)  # pumps the window's events, including the zoom double-click
                next_frame += period
                self.stop_event.wait(max(0.0, next_frame - time.monotonic()))
                next_frame = max(next_frame, time.monotonic())
        finally:
            ring.display_fps = 0
            cv2.destroyAllWindows()
            ring.close()

    def _open_window(self, renderer: GridRenderer) -> bool:
        h, w = renderer.canvas.shape[:2]
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, w, h)
        cv2.moveWindow(WINDOW, 0, 0)

        def on_mouse(event: int, x: int, y: int, *_) -> None:
            if event == cv2.EVENT_LBUTTONDBLCLK:
                renderer.toggle_zoom(x, y)

        cv2.setMouseCallback(WINDOW, on_mouse)
        cv2.waitKey(1)  # the native window exists only after the first event pump
        hwnd = windows.find_hwnd(WINDOW)
        windows.disable_close_button(hwnd, tag=self.name)  # it is closed from the operator panel
        windows.force_foreground(hwnd, tag=self.name)
        return True

    def _keep_masks(self) -> dict[int, np.ndarray]:
        masks = RegionMasks.from_dir(self.mask_dir, self.macs, self.layout.frame_size, name=self.name)
        return {} if masks is None else {sid: masks.keep[sid] for sid in masks.covered}


class MonitorHandler(Handler):
    """Publishes each batch's per-camera status and runs the monitor on demand."""

    name = "Monitor"

    def __init__(self, ring: FrameRing, *, enabled: bool, audit: AuditLog = NULL_AUDIT, **process_kwargs):
        super().__init__(audit)
        self.ring = ring
        self.process_kwargs = process_kwargs
        self._start_enabled = enabled
        self._process: MonitorProcess | None = None

    def start(self) -> None:
        if self._start_enabled:
            self.set_enabled(True)

    def stop(self) -> None:
        self.set_enabled(False)

    def handle(self, batch: Batch) -> None:
        r = batch.records
        status = self.ring.status
        sids = r["sid"]
        status["frame_idx"][sids] = r["frame_idx"]
        status["count"][sids] = r["count"]
        status["slot"][sids] = r["slot"]
        status["level"][sids] = [0 if d is None else d.level for d in batch.decisions]
        status["alert"][sids] = [batch.alert(i) for i in range(len(r))]

    # Toggles (operator / debug panel) -----------------------------------------

    def is_enabled(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def set_enabled(self, on: bool) -> None:
        if on and not self.is_enabled():
            self._process = MonitorProcess(self.ring.layout, **self.process_kwargs)
            self._process.start()
            self.audit.log("monitor.enable")
            logger.info("Monitor on.")
        elif not on and self._process is not None:
            proc, self._process = self._process, None
            proc.stop()
            proc.join(timeout=_TERMINATE_GRACE)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=_TERMINATE_GRACE)
            self.ring.display_fps = 0  # in case the process could not reset it itself
            self.audit.log("monitor.disable")
            logger.info("Monitor off.")

    def set_overlay(self, name: str, on: bool) -> None:
        """`density` or `mask`; read live by the monitor process."""
        setattr(self.ring, f"show_{name}", on)
        self.audit.log(f"monitor.{name}", enabled=on)
