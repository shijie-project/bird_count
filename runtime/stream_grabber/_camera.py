import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from runtime.config import Config
from runtime.shared_memory import BufferState


_RECONNECT_BACKOFF_BASE = 2.0  # seconds, doubled per failure
_RECONNECT_BACKOFF_MAX = 30.0  # cap on the exponential
_POST_FAILURE_BACKOFF = 2.0  # short pause after a mid-stream grab failure

logger = logging.getLogger(__name__)


class CameraThread(threading.Thread):
    """
    Per-stream capture thread.

    Owns one `cv2.VideoCapture` and writes decoded frames into the stream's
    ring of SHM slots. Reconnects with exponential backoff on failure; emits
    a cached 'NO SIGNAL' frame between connection attempts so downstream
    consumers always have something fresh to read.

    Uses a `threading.Event` for shutdown so every wait is interruptible —
    `stop()` returns control within ~one frame interval, not up to 30s of
    reconnect backoff.
    """

    def __init__(
        self,
        stream_id: int,
        source: str,
        metadata: np.ndarray,
        frames: np.ndarray,
        config: Config,
    ):
        super().__init__(name=f"CamThread-{stream_id}", daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.config = config

        # Per-stream SHM views, sliced by the parent process and passed in
        # directly. The thread never sees other streams' data — this is the
        # invariant the narrow signature enforces.
        self.metadata = metadata
        self.frames = frames
        self.num_buffers = config.num_buffers

        # cv2.resize wants (W, H); numpy shape compares against (H, W). Keep both.
        self._target_wh = (config.shm.width, config.shm.height)
        self._target_hw = (config.shm.height, config.shm.width)

        # `_interval` is the target frame period. For a live camera it's the
        # configured FPS (we only want the freshest frame anyway). For a video
        # file we override it per-capture with the clip's *native* FPS so it
        # plays back at real-time speed instead of being slowed to the
        # inference FPS — `_active_interval` holds the value actually in use.
        self._interval = config.frame_interval
        self._is_video = config.envs.source_type == "video"
        self._active_interval = self._interval

        self._stop_event = threading.Event()
        self._frame_idx = 0
        self._consecutive_connect_failures = 0
        # Guards video looping: set after a rewind so a genuinely unreadable
        # file falls through to a reconnect instead of spinning on seek.
        self._rewind_guard = False

        # Render the 'NO SIGNAL' tile once; reuse across reconnect attempts.
        self._no_signal_frame = self._build_no_signal_frame()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self):
        while not self._stop_event.is_set():
            # Keep downstream fed while we attempt to (re)connect.
            self._write_no_signal()

            cap = self._open_capture()
            if cap is None:
                self._backoff_after_connect_failure()
                continue

            try:
                self._capture_loop(cap)
            finally:
                cap.release()

            # Only reached when the inner loop bailed; pause briefly before
            # the next reconnect so we don't hammer a dead source.
            self._stop_event.wait(timeout=_POST_FAILURE_BACKOFF)

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        # FFmpeg backend gives consistent codec behavior across platforms.
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        # Smallest possible internal buffer = lowest end-to-end latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        self._active_interval = self._resolve_interval(cap)
        self._rewind_guard = False
        logger.info("[%s] Connected (%.1f FPS playback).", self.name, 1.0 / self._active_interval)
        self._consecutive_connect_failures = 0
        return cap

    def _resolve_interval(self, cap: cv2.VideoCapture) -> float:
        """Frame period to pace this capture at.

        Video file → the clip's native FPS (real-time playback). Live camera,
        or a video with no usable FPS metadata → the configured interval.
        """
        if self._is_video:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            # Guard against 0 / NaN / absurd metadata from odd containers.
            if native_fps and 0.0 < native_fps <= 240.0:
                return 1.0 / native_fps
        return self._interval

    def _rewind(self, cap: cv2.VideoCapture) -> bool:
        """Seek a video source back to frame 0 for seamless looping.

        Returns True if the seek succeeded and the caller should retry grabbing;
        False (source genuinely dead, or we already rewound once without
        recovering) to fall through to the reconnect path.
        """
        if self._rewind_guard:
            return False
        self._rewind_guard = True
        return bool(cap.set(cv2.CAP_PROP_POS_FRAMES, 0))

    def _backoff_after_connect_failure(self):
        self._consecutive_connect_failures += 1
        wait = min(
            _RECONNECT_BACKOFF_BASE**self._consecutive_connect_failures,
            _RECONNECT_BACKOFF_MAX,
        )
        logger.error("[%s] Source open failed. Retrying in %.1fs.", self.name, wait)
        # Interruptible sleep — stop() unblocks immediately instead of
        # waiting out the full exponential backoff.
        self._stop_event.wait(timeout=wait)

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self, cap: cv2.VideoCapture):
        """Grab / retrieve / ingest.

        Returns on any failure so the outer loop reconnects — except that a
        video source that hits EOF is rewound in place so the clip loops
        seamlessly (no reconnect backoff, no 'NO SIGNAL' flash between loops).
        """
        interval = self._active_interval
        wait = self._stop_event.wait
        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            # Split grab/retrieve so decode cost is only paid when needed.
            if not cap.grab():
                if self._is_video and self._rewind(cap):
                    continue
                return
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                if self._is_video and self._rewind(cap):
                    continue
                return

            # A real frame decoded — clear the loop guard so the *next* EOF is
            # allowed to rewind again.
            self._rewind_guard = False
            self._ingest(frame)

            # Precise FPS throttle (interruptible).
            elapsed = time.perf_counter() - t_start
            if elapsed < interval:
                wait(timeout=interval - elapsed)

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def _ingest(self, frame: np.ndarray):
        """Resize-if-needed and write the frame into the next available SHM slot."""
        self._frame_idx += 1

        target_idx = self._select_target_slot()
        if target_idx is None:
            logger.debug("[%s] Congestion: no buffer available. Dropping #%d", self.name, self._frame_idx)
            return

        if frame.shape[:2] != self._target_hw:
            frame = cv2.resize(frame, self._target_wh, interpolation=cv2.INTER_LINEAR)

        self._write_slot(target_idx, frame)

    def _select_target_slot(self) -> Optional[int]:
        """
        Pick a buffer to write into:
          1. Prefer any FREE slot.
          2. Else, overwrite the oldest READY slot (drops a stale frame).
          3. Else, decline (all slots locked by WRITING / READING).
        """
        states = self.metadata["state"]

        free = np.flatnonzero(states == BufferState.FREE)
        if free.size:
            return int(free[0])

        ready = np.flatnonzero(states == BufferState.READY)
        if ready.size:
            # Overwrite the oldest READY slot.
            return int(ready[np.argmin(self.metadata["frame_idx"][ready])])

        return None

    def _write_slot(self, target_idx: int, frame: np.ndarray):
        target_meta = self.metadata[target_idx]
        try:
            # State machine: WRITING -> fill -> READY. Inference reads only
            # READY slots, so this sequence prevents partial-frame reads.
            target_meta["state"] = BufferState.WRITING
            np.copyto(self.frames[target_idx], frame)
            target_meta["frame_idx"] = self._frame_idx
            target_meta["timestamp"] = time.time()
            target_meta["stream_id"] = self.stream_id
            target_meta["buffer_idx"] = target_idx
            target_meta["state"] = BufferState.READY
        except Exception as e:
            logger.error("[%s] Write failed: %s", self.name, e, exc_info=True)
            target_meta["state"] = BufferState.FREE

    # ------------------------------------------------------------------
    # 'NO SIGNAL' tile
    # ------------------------------------------------------------------

    def _write_no_signal(self):
        """Push the cached 'NO SIGNAL' frame so downstream consumers don't starve."""
        self._ingest(self._no_signal_frame)

    def _build_no_signal_frame(self) -> np.ndarray:
        w, h = self._target_wh
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        text = "NO SIGNAL"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(
            frame,
            text,
            ((w - tw) // 2, (h + th) // 2),
            font,
            font_scale,
            (200, 200, 200),
            thickness,
        )
        return frame
