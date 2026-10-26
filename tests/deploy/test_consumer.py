"""The consumer: one decision per batch, every handler served, slots always returned; the camera watch."""

import time

import numpy as np

from birdcount.deploy.consumer import Consumer
from birdcount.deploy.gui.controls import ControlState
from birdcount.deploy.handlers.base import Handler
from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.results import empty_batch
from birdcount.deploy.shm import SlotState


class Recorder(Handler):
    name = "Rec"

    def __init__(self, fail=False):
        super().__init__()
        self.batches, self.fail, self.cancelled = [], fail, 0

    def handle(self, batch):
        self.batches.append(batch)
        if self.fail:
            raise RuntimeError("boom")

    def cancel_all(self):
        self.cancelled += 1

    def active_devices(self):
        return {"10.0.1.1"}


class Audit:
    def __init__(self):
        self.events = []

    def log(self, event, **fields):
        self.events.append((event, fields.get("stream_id")))


def claimed(ring, sids):
    """Publish and claim a frame per camera, as inference does; returns result records."""
    records = empty_batch(len(sids))
    for i, sid in enumerate(sids):
        slot = ring.begin_write(sid)
        ring.publish(sid, slot, 1, time.time())
        ring.begin_read(sid, slot)
        records[i]["sid"], records[i]["slot"], records[i]["count"] = sid, slot, 10.0
    return records


def consumer(ring, handlers, **kw):
    return Consumer(ring, AlarmEngine(None, [None, None]), handlers, ControlState(2), labels=["a", "b"], **kw)


def test_every_handler_sees_the_batch_and_slots_go_back_even_if_one_fails(ring) -> None:
    bad, good = Recorder(fail=True), Recorder()
    c = consumer(ring, [bad, good])
    c.control.toggle(1)
    records = claimed(ring, [0, 1])
    c.process(records)
    assert len(bad.batches) == len(good.batches) == 1
    batch = good.batches[0]
    assert batch.manual == {1} and batch.decisions == [None, None]  # no alarm config: hijack is device-only
    assert (ring.meta["state"][[0, 1], records["slot"]] == SlotState.FREE).all()
    assert c.batches == 1


def test_cancel_all_clears_hijacks_and_reaches_every_handler(ring) -> None:
    handlers = [Recorder(), Recorder(fail=True)]
    audit = Audit()
    c = consumer(ring, handlers, audit=audit)
    c.control.toggle(0)
    c.cancel_all()
    assert c.control.hijacked == set() and [h.cancelled for h in handlers] == [1, 1]
    assert ("alert.cancel_all", None) in audit.events
    assert c.active_devices() == {"Rec": {"10.0.1.1"}}


def test_camera_watch_reports_a_camera_down_once_then_back(ring) -> None:
    audit = Audit()
    c = consumer(ring, [], offline_after=30.0, audit=audit)
    ring.status["online"] = [1, 0]
    c.watch_cameras(100.0)
    c.watch_cameras(120.0)
    assert audit.events == []  # not yet 30 s
    c.watch_cameras(131.0)
    c.watch_cameras(140.0)
    assert audit.events == [("camera.offline", 1)]
    ring.status["online"][1] = 1
    c.watch_cameras(141.0)
    assert audit.events == [("camera.offline", 1), ("camera.online", 1)]
    ring.status["online"][1] = 0  # a short drop is not reported
    c.watch_cameras(150.0)
    ring.status["online"][1] = 1
    c.watch_cameras(151.0)
    assert len(audit.events) == 2
    assert c.offline() == set()
    np.testing.assert_array_equal(ring.status["online"], [1, 1])
