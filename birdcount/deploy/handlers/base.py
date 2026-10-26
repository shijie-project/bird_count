"""The handler contract and the batch every handler receives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from birdcount.deploy.alarm import AlarmAction, CountSample
from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.shm import FrameRing


@dataclass(frozen=True)
class Decision:
    """The alarm engine's verdict on one result from a camera the alarm covers."""

    camera_id: str  # the alarm config's id, `axisN/MAC`
    sample: CountSample  # what the decision was made on (count, motion, simulation)
    actions: tuple[AlarmAction, ...]  # level1 / level2 / level3 / recovery emitted by this sample
    level: int  # level of the event open after this sample; 0 = none
    event_id: str | None  # that event
    simulated: bool  # a debug manual trigger posing as a pile-up


@dataclass(frozen=True)
class Batch:
    """One inference batch, as every handler sees it.

    `records` are `results.RESULT_DTYPE` rows; `decisions[i]` belongs to
    `records[i]` (None: camera not covered by the alarm). `manual` holds the
    cameras whose zone devices are switched on by hand, without an alarm.
    Frames are readable through `frame(i)` until `handle` returns; keep a
    `.copy()` for anything used later.
    """

    records: np.ndarray
    decisions: Sequence[Decision | None]
    manual: frozenset[int]
    ring: FrameRing | None = None

    def frame(self, i: int) -> np.ndarray:
        r = self.records[i]
        return self.ring.frames[int(r["sid"]), int(r["slot"])]

    def alert(self, i: int) -> bool:
        """An alarm event is open on this camera, or it is hijacked by hand."""
        d = self.decisions[i]
        return int(self.records[i]["sid"]) in self.manual or (d is not None and d.level > 0)


class Handler:
    """Base class: override `handle`; the rest is optional.

    `handle` runs on the consumer thread and must not block on I/O.
    """

    name = "Handler"

    def __init__(self, audit: AuditLog = NULL_AUDIT):
        self.audit = audit

    def start(self) -> None:
        """Acquire threads, files, clients."""

    def stop(self) -> None:
        """Release them; leave devices switched off."""

    def handle(self, batch: Batch) -> None:
        raise NotImplementedError

    def cancel_all(self) -> None:
        """Operator "cancel all": drop every in-flight alarm action."""

    def active_devices(self) -> set[str]:
        """Devices (or cameras) this handler currently has on, for status badges."""
        return set()
