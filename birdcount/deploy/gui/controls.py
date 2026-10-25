"""What the panels act on, with no Tk in it."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from birdcount.deploy.audit import NULL_AUDIT, AuditLog


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Toggle:
    """Any on/off switch a panel shows: the monitor, an overlay, recording."""

    label: str
    get: Callable[[], bool]
    set: Callable[[bool], None]


class ControlState:
    """Cameras hijacked by hand from the debug panel.

    The single copy: the debug grid draws from it and the alarm engine reads
    it every batch (v2 kept a second set in the grid widget and re-synced the
    two by hand). Tk runs on the consumer thread, so no lock is needed.
    """

    def __init__(self, num_streams: int, audit: AuditLog = NULL_AUDIT):
        self.num_streams = num_streams
        self.audit = audit
        self.hijacked: set[int] = set()

    def toggle(self, sid: int) -> bool:
        on = sid not in self.hijacked
        if on:
            self.hijacked.add(sid)
        else:
            self.hijacked.discard(sid)
        self.audit.log("hijack.enable" if on else "hijack.disable", stream_id=sid)
        logger.info("Manual hijack %s for camera %d.", "ON" if on else "OFF", sid)
        return on

    def set_all(self, on: bool) -> None:
        everyone = set(range(self.num_streams))
        changed = sorted(everyone - self.hijacked if on else self.hijacked)
        self.hijacked = everyone if on else set()
        for sid in changed:
            self.audit.log("hijack.enable" if on else "hijack.disable", stream_id=sid, source="trigger_all")
        logger.info("Manual hijack %s on %d camera(s).", "ON" if on else "OFF", len(changed))

    def clear(self) -> list[int]:
        cleared, self.hijacked = sorted(self.hijacked), set()
        return cleared
