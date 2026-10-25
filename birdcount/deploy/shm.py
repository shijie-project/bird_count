"""The shared-memory frame ring connecting grabber, inference and consumer.

Per camera, `num_buffers` slots of (frame, density, metadata), plus one
control record shared by everyone:

    frames    (S, B, H, W, 3) uint8 BGR   written by the grabber
    density   (S, B, H/8, W/8) float16    written by inference, read by the monitor
    meta      (S, B) {state, frame_idx, timestamp}
    control   {t0, interval, display_all} the inference clock, and whether
              the monitor wants every frame

Slot life cycle:

    FREE --begin_write--> WRITING --publish--> READY --begin_read--> READING --release--> FREE
                                                 ^ the grabber may overwrite the oldest READY slot

Torn reads are excluded by a seqlock on `frame_idx`: a writer sets it to -1
before touching pixels, and a reader only accepts a frame whose `frame_idx`
is unchanged after its copy (`verify`). That covers the one race the state
machine alone cannot: the grabber and the reader claiming the same last
READY slot at the same instant.

numpy only (no torch): every process attaches.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import shared_memory

import numpy as np


logger = logging.getLogger(__name__)


class SlotState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    READING = 3


META_DTYPE = np.dtype([("state", np.uint8), ("frame_idx", np.int64), ("timestamp", np.float64)], align=True)
CONTROL_DTYPE = np.dtype([("t0", np.float64), ("interval", np.float64), ("display_all", np.uint8)], align=True)


@dataclass(frozen=True)
class RingLayout:
    """Everything needed to allocate or attach; picklable across spawn."""

    prefix: str
    num_streams: int
    num_buffers: int
    frame_size: tuple[int, int]  # (width, height)
    stride: int = 8

    def blocks(self) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        w, h = self.frame_size
        s, b = self.num_streams, self.num_buffers
        return {
            "frames": ((s, b, h, w, 3), np.dtype(np.uint8)),
            "density": ((s, b, h // self.stride, w // self.stride), np.dtype(np.float16)),
            "meta": ((s, b), META_DTYPE),
            "control": ((), CONTROL_DTYPE),
        }

    @property
    def size_bytes(self) -> int:
        return sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in self.blocks().values())


class FrameRing:
    """One process's handle on the ring. `allocate` in the owner, `attach` elsewhere."""

    def __init__(self, layout: RingLayout, segments: dict[str, shared_memory.SharedMemory], owner: bool):
        self.layout = layout
        self._segments = segments
        self._owner = owner
        arrays = {
            name: np.ndarray(shape, dtype=dtype, buffer=segments[name].buf)
            for name, (shape, dtype) in layout.blocks().items()
        }
        self.frames: np.ndarray = arrays["frames"]
        self.density: np.ndarray = arrays["density"]
        self.meta: np.ndarray = arrays["meta"]
        self.control: np.ndarray = arrays["control"]

    @classmethod
    def allocate(cls, layout: RingLayout, *, fps: float) -> FrameRing:
        segments = {}
        try:
            for name, (shape, dtype) in layout.blocks().items():
                size = max(1, int(np.prod(shape)) * dtype.itemsize)
                segments[name] = shared_memory.SharedMemory(name=f"{layout.prefix}_{name}", create=True, size=size)
        except BaseException:
            for seg in segments.values():
                seg.close()
                seg.unlink()
            raise
        ring = cls(layout, segments, owner=True)
        ring.meta["state"] = SlotState.FREE
        ring.meta["frame_idx"] = 0
        ring.meta["timestamp"] = 0.0
        ring.control["t0"] = time.monotonic()
        ring.control["interval"] = 1.0 / fps
        ring.control["display_all"] = 0
        logger.info(
            "Frame ring %s: %.1f MB for %d streams x %d slots.",
            layout.prefix,
            layout.size_bytes / 2**20,
            layout.num_streams,
            layout.num_buffers,
        )
        return ring

    @classmethod
    def attach(cls, layout: RingLayout) -> FrameRing:
        segments = {name: shared_memory.SharedMemory(name=f"{layout.prefix}_{name}") for name in layout.blocks()}
        return cls(layout, segments, owner=False)

    def close(self) -> None:
        """Detach; the owner also destroys the segments."""
        self.frames = self.density = self.meta = self.control = None
        for seg in self._segments.values():
            seg.close()
            if self._owner:
                seg.unlink()
        self._segments = {}

    def __enter__(self) -> FrameRing:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    @property
    def interval(self) -> float:
        return float(self.control["interval"])

    def next_tick(self, now: float) -> float:
        """The first inference tick strictly after `now` (time.monotonic)."""
        t0, interval = float(self.control["t0"]), self.interval
        return t0 + (float(np.floor((now - t0) / interval)) + 1) * interval

    @property
    def display_all(self) -> bool:
        return bool(self.control["display_all"])

    @display_all.setter
    def display_all(self, value: bool) -> None:
        self.control["display_all"] = int(value)

    # ------------------------------------------------------------------
    # Writer side (grabber)
    # ------------------------------------------------------------------

    def begin_write(self, sid: int) -> int | None:
        """Claim a slot to write: any FREE one, else the oldest READY; None if all are busy."""
        meta = self.meta[sid]
        states = meta["state"]
        free = np.flatnonzero(states == SlotState.FREE)
        if free.size:
            slot = int(free[0])
        else:
            ready = np.flatnonzero(states == SlotState.READY)
            if not ready.size:
                return None
            slot = int(ready[np.argmin(meta["frame_idx"][ready])])
        meta["state"][slot] = SlotState.WRITING
        meta["frame_idx"][slot] = -1  # seqlock: invalidates any concurrent reader of this slot
        return slot

    def publish(self, sid: int, slot: int, frame_idx: int, timestamp: float) -> None:
        meta = self.meta[sid]
        meta["timestamp"][slot] = timestamp
        meta["frame_idx"][slot] = frame_idx
        meta["state"][slot] = SlotState.READY

    def abort_write(self, sid: int, slot: int) -> None:
        self.meta[sid]["state"][slot] = SlotState.FREE

    # ------------------------------------------------------------------
    # Reader side (inference)
    # ------------------------------------------------------------------

    def newest_ready(self, sid: int) -> tuple[int, int] | None:
        """(slot, frame_idx) of the newest READY frame, or None."""
        meta = self.meta[sid]
        ready = np.flatnonzero(meta["state"] == SlotState.READY)
        if not ready.size:
            return None
        slot = int(ready[np.argmax(meta["frame_idx"][ready])])
        return slot, int(meta["frame_idx"][slot])

    def begin_read(self, sid: int, slot: int) -> None:
        """Claim `slot` for reading and free every older READY slot (stale frames)."""
        meta = self.meta[sid]
        meta["state"][slot] = SlotState.READING
        stale = (meta["state"] == SlotState.READY) & (meta["frame_idx"] < meta["frame_idx"][slot])
        meta["state"][stale] = SlotState.FREE

    def verify(self, sid: int, slot: int, frame_idx: int) -> bool:
        """True if nothing overwrote the slot since `newest_ready` returned `frame_idx`."""
        meta = self.meta[sid]
        return int(meta["frame_idx"][slot]) == frame_idx and meta["state"][slot] == SlotState.READING

    def release(self, sids: np.ndarray, slots: np.ndarray) -> None:
        """Return read slots to the writer (consumer, once every handler is done).

        Only slots still READING are freed: one the grabber has meanwhile
        re-published holds a newer frame and stays READY.
        """
        states = self.meta["state"]
        still_read = states[sids, slots] == SlotState.READING
        states[sids[still_read], slots[still_read]] = SlotState.FREE
