"""MonitorHandler: per-camera status into the ring, overlay switches."""

import pytest

from birdcount.deploy.alarm import CountSample
from birdcount.deploy.audit import NULL_AUDIT
from birdcount.deploy.handlers.base import Batch, Decision
from birdcount.deploy.monitor.process import MonitorHandler
from birdcount.deploy.results import empty_batch
from birdcount.deploy.shm import FrameRing

from ..conftest import make_layout


@pytest.fixture
def ring():
    r = FrameRing.allocate(make_layout(streams=3), fps=10)
    yield r
    r.close()


def decision(level: int) -> Decision:
    return Decision("axis1/X", CountSample(0.0, "axis1/X", 1.0), (), level, "e" if level else None, False)


def test_status_follows_each_batch(ring) -> None:
    handler = MonitorHandler(ring, enabled=False)
    r = empty_batch(3)
    r["sid"], r["slot"], r["frame_idx"], r["count"] = [2, 0, 1], [3, 1, 0], [7, 8, 9], [12.5, 140.0, 3.0]
    handler.handle(Batch(r, [None, decision(2), decision(0)], manual=frozenset({2})))
    status = ring.status
    assert status["frame_idx"].tolist() == [8, 9, 7]
    assert status["count"].tolist() == [140.0, 3.0, 12.5]
    assert status["slot"].tolist() == [1, 0, 3]
    assert status["level"].tolist() == [2, 0, 0]
    assert status["alert"].tolist() == [1, 0, 1]  # camera 0: alarm open; camera 2: hijacked


def test_overlays_and_the_monitor_switch(ring) -> None:
    handler = MonitorHandler(ring, enabled=False, audit=NULL_AUDIT)
    handler.set_overlay("density", True)
    handler.set_overlay("mask", True)
    assert ring.show_density and ring.show_mask
    handler.set_overlay("mask", False)
    assert not ring.show_mask
    assert not handler.is_enabled()
    handler.stop()  # stopping a monitor that never ran is a no-op
    assert ring.display_fps == 0
