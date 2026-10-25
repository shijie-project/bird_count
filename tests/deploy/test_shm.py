import time

import numpy as np
import pytest

from birdcount.deploy.shm import FrameRing, SlotState

from .conftest import make_layout


def _write(ring, sid, idx) -> int:
    slot = ring.begin_write(sid)
    ring.frames[sid, slot] = idx
    ring.publish(sid, slot, idx, time.time())
    return slot


def test_attach_sees_the_owners_data(ring) -> None:
    other = FrameRing.attach(ring.layout)
    try:
        slot = _write(ring, 1, 7)
        assert other.frames[1, slot].min() == 7
        assert other.newest_ready(1) == (slot, 7)
        assert other.interval == pytest.approx(0.1)
    finally:
        other.close()


def test_clock_ticks_are_on_the_shared_grid(ring) -> None:
    t0, interval = float(ring.control["t0"]), ring.interval
    tick = ring.next_tick(t0 + 2.35 * interval)
    assert tick == pytest.approx(t0 + 3 * interval)
    assert ring.next_tick(tick) == pytest.approx(t0 + 4 * interval)


def test_writer_prefers_free_then_oldest_ready_then_gives_up(ring) -> None:
    slots = [_write(ring, 0, i) for i in range(1, 5)]
    assert sorted(slots) == [0, 1, 2, 3]
    assert ring.begin_write(0) == slots[0]  # oldest READY is overwritten
    ring.meta["state"][0] = SlotState.READING
    assert ring.begin_write(0) is None


def test_reading_the_newest_frees_stale_frames(ring) -> None:
    for i in range(1, 4):
        _write(ring, 0, i)
    slot, idx = ring.newest_ready(0)
    assert idx == 3
    ring.begin_read(0, slot)
    states = ring.meta["state"][0]
    assert states[slot] == SlotState.READING
    assert (states == SlotState.FREE).sum() == 3
    assert ring.verify(0, slot, idx)


def test_seqlock_rejects_a_frame_overwritten_during_the_read(ring) -> None:
    ring.meta["state"][0] = SlotState.READING  # no FREE slots left...
    ring.meta["state"][0, 2] = SlotState.READY  # ...and a single READY one
    ring.meta["frame_idx"][0, 2] = 5
    slot, idx = ring.newest_ready(0)
    assert ring.begin_write(0) == slot  # the grabber grabs the same slot
    ring.begin_read(0, slot)  # and the reader claims it too
    assert not ring.verify(0, slot, idx)


def test_release_leaves_republished_slots_alone(ring) -> None:
    a = _write(ring, 0, 1)
    b = _write(ring, 1, 1)
    ring.begin_read(0, a)
    ring.begin_read(1, b)
    ring.publish(1, b, 2, time.time())  # re-published while held
    ring.release(np.array([0, 1]), np.array([a, b]))
    assert ring.meta["state"][0, a] == SlotState.FREE
    assert ring.meta["state"][1, b] == SlotState.READY


def test_display_rate_and_overlay_flags(ring) -> None:
    other = FrameRing.attach(ring.layout)
    try:
        assert other.display_fps == 0 and not other.show_density and not other.show_mask
        ring.display_fps, ring.show_density, ring.show_mask = 25, True, True
        assert other.display_fps == pytest.approx(25) and other.show_density and other.show_mask
    finally:
        other.close()


def test_frames_are_due_on_the_display_grid_while_it_is_open(ring) -> None:
    t0, interval = float(ring.control["t0"]), ring.interval  # 0.1 s inference ticks
    now = t0 + 0.01
    assert ring.next_due(now) == pytest.approx(t0 + interval)
    ring.display_fps = 40  # 0.025 s display ticks
    assert ring.next_due(now) == pytest.approx(t0 + 0.025)


def test_latest_sees_frames_being_read(ring) -> None:
    a = _write(ring, 0, 1)
    b = _write(ring, 0, 2)
    ring.begin_read(0, b)
    assert ring.latest(0) == (b, 2)
    assert ring.latest(1) is None
    assert ring.meta["state"][0, a] == SlotState.FREE


def test_names_are_unique_per_ring() -> None:
    a = FrameRing.allocate(make_layout(), fps=5)
    try:
        with pytest.raises(FileExistsError):
            FrameRing.allocate(a.layout, fps=5)
    finally:
        a.close()
