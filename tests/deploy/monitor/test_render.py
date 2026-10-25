"""The monitor grid: what each tile shows, and that only changed tiles are redrawn."""

import time

import numpy as np
import pytest

from birdcount.deploy.monitor.render import GridRenderer
from birdcount.deploy.shm import FrameRing

from ..conftest import H, W, make_layout


CANVAS = (384, 216)  # two cameras -> two 192x216 tiles side by side
GRAY = 120


@pytest.fixture
def ring():
    r = FrameRing.allocate(make_layout(streams=2), fps=10)
    yield r
    r.close()


def publish(ring: FrameRing, sid: int, idx: int, gray: int = GRAY) -> int:
    slot = ring.begin_write(sid)
    ring.frames[sid, slot] = gray
    ring.publish(sid, slot, idx, time.time())
    return slot


def bottom_right(renderer: GridRenderer, sid: int) -> np.ndarray:
    """A pixel clear of the header and the border."""
    r, c = divmod(sid, renderer.cols)
    return renderer.canvas[(r + 1) * renderer.tile_h - 10, (c + 1) * renderer.tile_w - 10]


def test_tiles_show_their_camera_and_only_changes_are_redrawn(ring) -> None:
    renderer = GridRenderer(2, CANVAS)
    assert not renderer.render(ring, flash_on=False)  # nothing published yet
    publish(ring, 0, 1, gray=50)
    publish(ring, 1, 1, gray=200)
    assert renderer.render(ring, flash_on=False)
    assert renderer.redraws == 2
    assert tuple(bottom_right(renderer, 0)) == (50, 50, 50)
    assert tuple(bottom_right(renderer, 1)) == (200, 200, 200)

    assert not renderer.render(ring, flash_on=False)
    assert renderer.redraws == 2

    publish(ring, 1, 2)  # a newer frame
    renderer.render(ring, flash_on=False)
    assert renderer.redraws == 3
    ring.status["count"][0] = 17.2  # a new count
    renderer.render(ring, flash_on=False)
    assert renderer.redraws == 4
    ring.status["count"][0] = 17.4  # the same count, as drawn
    renderer.render(ring, flash_on=False)
    assert renderer.redraws == 4


def test_tile_hit_testing_and_zoom(ring) -> None:
    renderer = GridRenderer(3, (300, 200))  # 2x2 grid with one empty cell
    assert renderer.tile_at(10, 10) == 0
    assert renderer.tile_at(160, 10) == 1
    assert renderer.tile_at(10, 110) == 2
    assert renderer.tile_at(160, 110) is None
    assert renderer.tile_at(-1, 0) is None

    renderer = GridRenderer(2, CANVAS)
    publish(ring, 0, 1, gray=50)
    publish(ring, 1, 1, gray=200)
    renderer.toggle_zoom(300, 100)  # into camera 1
    assert renderer.focused == 1
    renderer.render(ring, flash_on=False)
    assert tuple(renderer.canvas[-10, 10]) == (200, 200, 200)  # left half is camera 1 now
    renderer.toggle_zoom(0, 0)
    assert renderer.focused is None
    renderer.render(ring, flash_on=False)
    assert tuple(bottom_right(renderer, 0)) == (50, 50, 50)


def test_alert_flashes_the_header(ring) -> None:
    renderer = GridRenderer(2, CANVAS)
    publish(ring, 0, 1)
    ring.status["alert"][0] = 1
    renderer.render(ring, flash_on=True)
    assert tuple(renderer.canvas[5, 5]) == (0, 0, 255)
    renderer.render(ring, flash_on=False)
    assert tuple(renderer.canvas[5, 5]) == (0, 0, 0)


def test_density_overlay_uses_the_latest_inferred_slot(ring) -> None:
    renderer = GridRenderer(2, CANVAS)
    slot = publish(ring, 0, 1)
    ring.density[0, slot] = 0.0
    ring.density[0, slot, -1, -1] = 1.0  # one hot cell in the bottom-right corner
    ring.show_density = True
    renderer.render(ring, flash_on=False)
    assert tuple(bottom_right(renderer, 0)) == (GRAY,) * 3  # no density yet: plain frame
    ring.status["slot"][0], ring.status["frame_idx"][0] = slot, 1
    renderer.render(ring, flash_on=False)
    assert tuple(bottom_right(renderer, 0)) != (GRAY,) * 3
    assert tuple(renderer.canvas[100, 20]) == (GRAY,) * 3  # cold cells stay uncoloured


def test_mask_overlay_tints_the_ignored_area_or_says_there_is_none(ring) -> None:
    keep = np.zeros((H, W), np.uint8)
    keep[:, : W // 2] = 1  # count the left half only
    renderer = GridRenderer(2, CANVAS, keep_masks={0: keep})
    publish(ring, 0, 1)
    publish(ring, 1, 1)
    ring.show_mask = True
    renderer.render(ring, flash_on=False)
    assert tuple(renderer.canvas[150, 20]) == (GRAY,) * 3  # kept
    assert tuple(bottom_right(renderer, 0)) != (GRAY,) * 3  # ignored: tinted
    no_mask = renderer.canvas[180:216, 192 + 10 : 192 + 120]
    assert (no_mask == (0, 0, 255)).all(axis=-1).any()  # camera 1 has no mask: red label


def test_header_follows_the_count(ring) -> None:
    renderer = GridRenderer(2, CANVAS)
    publish(ring, 0, 1)
    ring.status["count"][0] = 7
    renderer.render(ring, flash_on=False)
    before = renderer.canvas[:40, :150].copy()
    ring.status["count"][0] = 140
    renderer.render(ring, flash_on=False)
    assert (renderer.canvas[:40, :150] != before).any()
    ring.status["count"][0] = 7
    renderer.render(ring, flash_on=False)
    np.testing.assert_array_equal(renderer.canvas[:40, :150], before)
