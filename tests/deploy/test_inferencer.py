import time

import numpy as np
import pytest
import torch
from torch import nn

from birdcount.core.inference import DensityPredictor
from birdcount.core.masks import RegionMasks
from birdcount.core.model import DensityNet
from birdcount.deploy.inferencer import InferenceLoop
from birdcount.deploy.shm import FrameRing, SlotState

from .conftest import H, W, make_layout


@pytest.fixture(scope="module")
def model() -> DensityNet:
    torch.manual_seed(0)
    m = DensityNet().eval()
    for bn in (x for x in m.modules() if isinstance(x, nn.BatchNorm2d)):
        bn.running_var.uniform_(0.5, 1.5)
    m.density_conv.bias.data.fill_(0.3)
    m.fuse_conv_bn()
    m.set_input_order("bgr")
    return m.requires_grad_(False)


@pytest.fixture
def ring3():
    r = FrameRing.allocate(make_layout(streams=3), fps=20)
    yield r
    r.close()


def _publish(ring, sid, frame, idx):
    slot = ring.begin_write(sid)
    ring.frames[sid, slot] = frame
    ring.publish(sid, slot, idx, time.time())
    return slot


def _frames(n):
    return np.random.default_rng(0).integers(0, 256, (n, H, W, 3), dtype=np.uint8)


def test_a_batch_matches_the_predictor(ring3, model) -> None:
    frames = _frames(3)
    slots = [_publish(ring3, s, frames[s], 1) for s in range(3)]
    predictor = DensityPredictor(model, frame_size=(W, H), max_batch=3)
    loop = InferenceLoop(ring3, predictor, [0, 1, 2], max_wait=0.5)
    batch = loop.step(time.monotonic() + 0.5)

    expected = DensityPredictor(model, frame_size=(W, H), max_batch=3).predict(frames, want_density=True)
    assert sorted(batch["sid"]) == [0, 1, 2]
    order = np.argsort(batch["sid"])
    np.testing.assert_allclose(batch["count"][order], expected.counts, rtol=1e-5)
    np.testing.assert_array_equal(batch["slot"][order], slots)
    for s in range(3):
        np.testing.assert_allclose(ring3.density[s, slots[s]], expected.density[s], rtol=1e-3, atol=1e-4)
        assert ring3.meta["state"][s, slots[s]] == SlotState.READING  # held until the consumer releases
    assert (batch["inferred"] >= batch["captured"]).all()


def test_frames_are_only_inferred_once(ring3, model) -> None:
    for s in range(3):
        _publish(ring3, s, _frames(1)[0], 1)
    loop = InferenceLoop(ring3, DensityPredictor(model, frame_size=(W, H), max_batch=3), [0, 1, 2], max_wait=0.05)
    assert len(loop.step(time.monotonic() + 0.05)) == 3
    assert loop.step(time.monotonic() + 0.05) is None


def test_a_silent_camera_is_not_waited_for(ring3, model) -> None:
    _publish(ring3, 0, _frames(1)[0], 1)
    _publish(ring3, 1, _frames(1)[0], 1)
    loop = InferenceLoop(ring3, DensityPredictor(model, frame_size=(W, H), max_batch=3), [0, 1, 2], max_wait=5.0)
    loop.last_fresh[2] = time.monotonic() - 10  # camera 2 has been silent for a while
    t = time.monotonic()
    batch = loop.step(t + 5.0)
    assert sorted(batch["sid"]) == [0, 1] and time.monotonic() - t < 1.0


def test_a_live_camera_is_waited_for_until_the_deadline(ring3, model) -> None:
    _publish(ring3, 0, _frames(1)[0], 1)
    loop = InferenceLoop(ring3, DensityPredictor(model, frame_size=(W, H), max_batch=3), [0, 1], max_wait=0.1)
    t = time.monotonic()
    batch = loop.step(t + 0.1)  # under the 3-tick (0.15 s) stall threshold: camera 1 is still live
    assert list(batch["sid"]) == [0] and time.monotonic() - t >= 0.09


def test_overwritten_frames_are_dropped(ring3, model, monkeypatch) -> None:
    for s in range(3):
        _publish(ring3, s, _frames(1)[0], 1)
    loop = InferenceLoop(ring3, DensityPredictor(model, frame_size=(W, H), max_batch=3), [0, 1, 2], max_wait=0.1)
    real = ring3.verify
    monkeypatch.setattr(ring3, "verify", lambda sid, slot, idx: sid != 1 and real(sid, slot, idx))
    batch = loop.step(time.monotonic() + 0.1)
    assert sorted(batch["sid"]) == [0, 2]


def test_mask_rows_follow_the_camera(ring3, model) -> None:
    frame = _frames(1)[0]
    for s in range(3):
        _publish(ring3, s, frame, 1)
    keep = np.ones((3, H, W), np.uint8)
    keep[1] = 0  # camera 1 counts nothing
    masks = RegionMasks(keep=keep, macs=("A", "B", "C"))
    loop = InferenceLoop(
        ring3,
        DensityPredictor(model, frame_size=(W, H), masks=masks, max_batch=3),
        [0, 1, 2],
        mask_rows=[0, 1, 2],
        max_wait=0.1,
    )
    batch = loop.step(time.monotonic() + 0.1)
    by_sid = dict(zip(batch["sid"], batch["count"], strict=True))
    assert by_sid[1] == 0 and by_sid[0] > 0 and by_sid[0] == pytest.approx(by_sid[2])
