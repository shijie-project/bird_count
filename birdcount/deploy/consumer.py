"""The main process's share of every inference batch, and the camera watch.

Per batch: the alarm engine decides once, every handler sees the same
immutable `Batch`, then the batch's ring slots go back to the grabber. A
handler that raises is logged and skipped; it never costs the other
handlers their batch, nor the grabber its slots.

The camera watch turns the grabber's per-camera `online` flag into one
WARNING + audit record when a camera has been down for `offline_after`
seconds, and one when it is back. v2 said nothing: a dead camera simply never
alarmed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.gui.controls import ControlState
from birdcount.deploy.handlers.base import Batch, Handler
from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.shm import FrameRing


logger = logging.getLogger(__name__)

OFFLINE_AFTER = 30.0  # s


class Consumer:
    def __init__(
        self,
        ring: FrameRing,
        engine: AlarmEngine,
        handlers: Sequence[Handler],
        control: ControlState,
        *,
        labels: Sequence[str],
        offline_after: float = OFFLINE_AFTER,
        audit: AuditLog = NULL_AUDIT,
    ):
        self.ring = ring
        self.engine = engine
        self.handlers = list(handlers)
        self.control = control
        self.labels = list(labels)
        self.offline_after = offline_after
        self.audit = audit
        self.batches = 0
        self._down_since: dict[int, float] = {}  # sid -> when it went down (or was first seen down)
        self._reported: set[int] = set()  # cameras reported offline

    def process(self, records: np.ndarray) -> Batch:
        """Decide, dispatch, release. Returns the batch for callers that inspect it."""
        try:
            decisions, manual = self.engine.evaluate(records, self.control.hijacked)
            batch = Batch(records, decisions, manual, self.ring)
            for handler in self.handlers:
                try:
                    handler.handle(batch)
                except Exception:
                    logger.exception("%s failed on a batch.", handler.name)
            self.batches += 1
            return batch
        finally:
            self.ring.release(records["sid"], records["slot"])

    # ------------------------------------------------------------------
    # Operator actions
    # ------------------------------------------------------------------

    def cancel_all(self) -> None:
        """Cancel All: clear hijacks, drop open alarm events, switch off what they switched on."""
        cleared = self.control.clear()
        self.audit.log("alert.cancel_all", cleared_hijacks=cleared)
        self.engine.reset_all()
        for handler in self.handlers:
            try:
                handler.cancel_all()
            except Exception:
                logger.exception("%s failed to cancel.", handler.name)
        logger.info("Cancel all: %d hijack(s) cleared.", len(cleared))

    def active_devices(self) -> dict[str, set[str]]:
        """Per handler, what it has switched on (devices, or cameras with an open SMS event)."""
        out = {}
        for handler in self.handlers:
            try:
                out[handler.name] = handler.active_devices()
            except Exception:
                out[handler.name] = set()
        return out

    # ------------------------------------------------------------------
    # Camera watch
    # ------------------------------------------------------------------

    def offline(self) -> set[int]:
        """Cameras the grabber currently has no picture from."""
        return set(np.flatnonzero(self.ring.status["online"] == 0).tolist())

    def watch_cameras(self, now: float) -> None:
        """Call about once a second (time.monotonic)."""
        down = self.offline()
        for sid in down:
            since = self._down_since.setdefault(sid, now)
            if sid not in self._reported and now - since >= self.offline_after:
                self._reported.add(sid)
                logger.warning(
                    "Camera %d (%s) has had no picture for %.0f s; it cannot alarm until it is back.",
                    sid,
                    self.labels[sid],
                    now - since,
                )
                self.audit.log("camera.offline", stream_id=sid, label=self.labels[sid])
        for sid in set(self._down_since) - down:
            del self._down_since[sid]
            if sid in self._reported:
                self._reported.discard(sid)
                logger.info("Camera %d (%s) is back.", sid, self.labels[sid])
                self.audit.log("camera.online", stream_id=sid, label=self.labels[sid])
