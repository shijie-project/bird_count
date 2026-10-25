"""Frame grabber: one decoding thread per camera, writing into the frame ring.

Every frame must be decoded (H.264 is inter-frame coded), but turning it into
a BGR image and copying it into shared memory is only done for frames that
will be used: the first frame decoded at or after each inference tick, or
the monitor's own frame rate while it is open (`FrameRing.next_due`). The BGR
conversion writes straight into the claimed ring slot — no intermediate
array. Measured on the demo clips: 0.97 ms of CPU per decoded frame with the
hardware decoder and tick-only retrieval, against v2's 4.06 ms.

Video files are paced at their own frame rate so they behave like cameras,
and loop at the end. A camera that drops is retried with exponential
backoff, and a "NO SIGNAL" frame is published while it is down, so the
consumer always has something current for every camera.

Torch-free: this process only needs numpy and cv2.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Sequence

import cv2
import numpy as np

from birdcount.core.log import setup_logging
from birdcount.deploy.shm import FrameRing, RingLayout
from birdcount.deploy.sources import Capture, Source, open_capture


logger = logging.getLogger(__name__)

_BACKOFF_BASE, _BACKOFF_MAX = 2.0, 30.0  # seconds; doubled per failed reconnect
_WATCHDOG_INTERVAL = 1.0


def no_signal_frame(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    frame = np.zeros((height, width, 3), np.uint8)
    text, font = "NO SIGNAL", cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 2.0 * width / 1280, max(1, round(3 * width / 1280))  # 2.0 / 3 at 720p
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, ((width - tw) // 2, (height + th) // 2), font, scale, (200, 200, 200), thick)
    return frame


class StreamReader(threading.Thread):
    """Decodes one source forever (until stopped) into its ring slots."""

    def __init__(
        self,
        source: Source,
        ring: FrameRing,
        *,
        hw_decode: bool,
        realtime: bool = True,
        backoff_base: float = _BACKOFF_BASE,
    ):
        super().__init__(name=f"Reader-{source.label}", daemon=True)
        self.source = source
        self.ring = ring
        self.hw_decode = hw_decode
        self.realtime = realtime  # pace files at their own rate; off only to measure capacity
        self.backoff_base = backoff_base
        self.stop_event = threading.Event()
        self.frame_idx = 0
        self.decoded = 0  # frames decoded, for tests and diagnostics
        w, h = ring.layout.frame_size
        self._size, self._shape = (w, h), (h, w, 3)
        self._no_signal = no_signal_frame((w, h))

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        failures = 0
        while not self.stop_event.is_set():
            self._publish(self._no_signal)
            capture = open_capture(self.source, hw_decode=self.hw_decode)
            if capture is None:
                failures += 1
                wait = min(self.backoff_base**failures, _BACKOFF_MAX)
                logger.error("[%s] Cannot open %s; retrying in %.0f s.", self.name, self.source.uri, wait)
                self.stop_event.wait(wait)
                continue
            failures = 0
            logger.info(
                "[%s] Opened (%s decode%s).",
                self.name,
                "hardware" if capture.hw_decode else "software",
                f", {capture.fps:.1f} fps file" if capture.fps else "",
            )
            try:
                self._read(capture)
            finally:
                capture.cap.release()
            self.stop_event.wait(min(self.backoff_base, _BACKOFF_MAX))

    def _read(self, capture: Capture) -> None:
        cap = capture.cap
        period = 1.0 / capture.fps if capture.fps and self.realtime else 0.0
        due = time.monotonic()
        rewound = False
        next_frame = time.perf_counter()
        while not self.stop_event.is_set():
            if not cap.grab():
                # End of a file: loop it. A source that fails right after a
                # rewind is genuinely broken, so reconnect instead.
                if self.source.is_file and not rewound and cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    rewound = True
                    continue
                return
            rewound = False
            self.decoded += 1

            now = time.monotonic()
            if now >= due:
                if not self._retrieve(cap):
                    return
                due = self.ring.next_due(now)

            if period:
                next_frame += period
                delay = next_frame - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                else:
                    next_frame = time.perf_counter()  # fell behind; don't try to catch up in a burst

    def _retrieve(self, cap: cv2.VideoCapture) -> bool:
        slot = self.ring.begin_write(self.source.sid)
        if slot is None:
            return True  # every slot busy: drop this frame, not the stream
        target = self.ring.frames[self.source.sid, slot]
        ok, frame = cap.retrieve(target)  # decodes straight into shared memory when the size matches
        if not ok or frame is None:
            self.ring.abort_write(self.source.sid, slot)
            return False
        if frame.shape != self._shape:
            cv2.resize(frame, self._size, dst=target, interpolation=cv2.INTER_LINEAR)  # core.frames.fit_frame's rule
        elif frame is not target:
            np.copyto(target, frame)
        self._commit(slot)
        return True

    def _publish(self, frame: np.ndarray) -> None:
        slot = self.ring.begin_write(self.source.sid)
        if slot is not None:
            np.copyto(self.ring.frames[self.source.sid, slot], frame)
            self._commit(slot)

    def _commit(self, slot: int) -> None:
        self.frame_idx += 1
        self.ring.publish(self.source.sid, slot, self.frame_idx, time.time())


class GrabberProcess(mp.Process):
    """Hosts a `StreamReader` per source and revives any that dies."""

    def __init__(
        self,
        layout: RingLayout,
        sources: Sequence[Source],
        *,
        hw_decode: bool = True,
        realtime: bool = True,
        log_level: str = "INFO",
    ):
        super().__init__(name="Grabber", daemon=True)
        self.layout = layout
        self.sources = list(sources)
        self.hw_decode = hw_decode
        self.realtime = realtime
        self.log_level = log_level
        self.stop_event = mp.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        setup_logging(self.log_level)
        ring = FrameRing.attach(self.layout)
        readers = [self._reader(s, ring) for s in self.sources]
        for r in readers:
            r.start()
        try:
            while not self.stop_event.wait(_WATCHDOG_INTERVAL):
                for i, r in enumerate(readers):
                    if not r.is_alive():
                        logger.warning("[%s] %s died; restarting it.", self.name, r.name)
                        readers[i] = self._reader(self.sources[i], ring)
                        readers[i].start()
        finally:
            for r in readers:
                r.stop()
            for r in readers:
                r.join(timeout=2.0)
            ring.close()

    def _reader(self, source: Source, ring: FrameRing) -> StreamReader:
        return StreamReader(source, ring, hw_decode=self.hw_decode, realtime=self.realtime)
