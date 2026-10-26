"""The inference pipeline's semantics, on CPU with a small random model.

What has to hold for every consumer to count the same way:
- the region mask makes ignored pixels irrelevant (they are seen as black),
- the density is masked before the sum, area-weighted,
- count = density sum and centroid = mass-weighted centre,
- normalisation matches (x/255 - mean)/std on the RGB image,
- batching does not change results.
"""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from birdcount.core.inference import IMAGENET_MEAN, IMAGENET_STD, DensityPredictor
from birdcount.core.masks import RegionMasks, density_weights
from birdcount.core.model import DensityNet


W, H = 96, 64
SIZE = (W, H)


@pytest.fixture(scope="module")
def model() -> DensityNet:
    torch.manual_seed(0)
    m = DensityNet().eval()
    g = torch.Generator().manual_seed(0)
    for bn in (x for x in m.modules() if isinstance(x, nn.BatchNorm2d)):
        bn.running_mean.normal_(0, 0.1, generator=g)
        bn.running_var.uniform_(0.5, 1.5, generator=g)
    m.density_conv.bias.data.fill_(0.3)  # keep the ReLU'd output non-zero
    m.fuse_conv_bn()
    m.set_input_order("bgr")
    return m.requires_grad_(False)


@pytest.fixture
def frames() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (5, H, W, 3), dtype=np.uint8)


@pytest.fixture
def masks() -> RegionMasks:
    keep = np.ones((2, H, W), np.uint8)
    keep[0, :, 52:] = 0  # boundary mid-cell (52 = 6.5 cells of 8)
    keep[1, :20] = 0
    return RegionMasks(keep=keep, macs=("A", "B"))


def _reference_density(model: DensityNet, frames_bgr: np.ndarray) -> torch.Tensor:
    """Textbook path: BGR -> RGB float, (x/255 - mean)/std, RGB model."""
    rgb = torch.from_numpy(frames_bgr[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255
    x = (rgb - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    model.set_input_order("rgb")
    try:
        with torch.no_grad():
            return model(x)
    finally:
        model.set_input_order("bgr")


def test_matches_the_textbook_pipeline(model, frames) -> None:
    pred = DensityPredictor(model, frame_size=SIZE, max_batch=8).predict(frames, want_density=True)
    ref = _reference_density(model, frames)
    np.testing.assert_allclose(pred.density, ref[:, 0].numpy(), rtol=1e-2, atol=2e-3)
    np.testing.assert_allclose(pred.counts, ref.sum(dim=(1, 2, 3)).numpy(), rtol=1e-3)


def test_count_is_the_sum_and_centroid_the_centre_of_mass(model, frames) -> None:
    pred = DensityPredictor(model, frame_size=SIZE, max_batch=8).predict(frames, want_density=True)
    d = pred.density.astype(np.float64)
    np.testing.assert_allclose(pred.counts, d.sum(axis=(1, 2)), rtol=1e-3)
    h, w = d.shape[1:]
    xs, ys = (np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h
    cx = (d.sum(axis=1) * xs).sum(axis=1) / d.sum(axis=(1, 2))
    cy = (d.sum(axis=2) * ys).sum(axis=1) / d.sum(axis=(1, 2))
    np.testing.assert_allclose(pred.centroids, np.stack([cx, cy], axis=1), atol=1e-4)


def test_ignored_pixels_do_not_matter(model, frames, masks) -> None:
    p = DensityPredictor(model, frame_size=SIZE, masks=masks, max_batch=8)
    scribbled = frames.copy()
    scribbled[:, :, 52:] = 255 - scribbled[:, :, 52:]  # change only what mask 0 ignores
    a = p.predict(frames, mask_ids=[0] * 5, want_density=True)
    b = p.predict(scribbled, mask_ids=[0] * 5, want_density=True)
    np.testing.assert_array_equal(a.density, b.density)


def test_masking_is_black_frame_then_weighted_density(model, frames, masks) -> None:
    p_masked = DensityPredictor(model, frame_size=SIZE, masks=masks, max_batch=8)
    p_plain = DensityPredictor(model, frame_size=SIZE, max_batch=8)
    black = frames * masks.keep[0][None, :, :, None]
    masked = p_masked.predict(frames, mask_ids=[0] * 5, want_density=True)
    plain = p_plain.predict(black, want_density=True)
    weights = density_weights(masks.keep[0], 8)
    assert ((weights > 0) & (weights < 1)).any()  # the boundary really is fractional
    np.testing.assert_allclose(masked.density, plain.density * weights, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(masked.counts, (plain.density * weights).sum(axis=(1, 2)), rtol=1e-3)


def test_per_frame_mask_rows(model, frames, masks) -> None:
    p = DensityPredictor(model, frame_size=SIZE, masks=masks, max_batch=8)
    mixed = p.predict(frames[:2], mask_ids=[1, 0])
    one = p.predict(frames[:1], mask_ids=[1])
    other = p.predict(frames[1:2], mask_ids=[0])
    np.testing.assert_allclose(mixed.counts, [one.counts[0], other.counts[0]], rtol=1e-5)


def test_batching_does_not_change_results(model, frames) -> None:
    big = DensityPredictor(model, frame_size=SIZE, max_batch=8).predict(frames)
    small = DensityPredictor(model, frame_size=SIZE, max_batch=2).predict(frames)
    np.testing.assert_allclose(small.counts, big.counts, rtol=1e-5)


def test_frames_are_fitted_to_the_model_geometry(model, frames) -> None:
    p = DensityPredictor(model, frame_size=SIZE, max_batch=8)
    large = np.stack([np.kron(f, np.ones((2, 2, 1), np.uint8)) for f in frames])  # 2x upscale
    assert p.predict(large).counts.shape == (5,)


def test_empty_density_has_no_centroid(model) -> None:
    p = DensityPredictor(model, frame_size=SIZE, masks=RegionMasks(np.zeros((1, H, W), np.uint8), ("A",)))
    pred = p.predict(np.zeros((1, H, W, 3), np.uint8), mask_ids=[0])
    assert pred.counts[0] == 0
    assert np.isnan(pred.centroids).all()


def test_staging_path_matches_predict(model, frames) -> None:
    p = DensityPredictor(model, frame_size=SIZE, max_batch=8)
    expected = p.predict(frames).counts
    p.staging[:5] = frames
    np.testing.assert_array_equal(p.run(5).counts, expected)


def test_rejects_misuse(model) -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        DensityPredictor(model, frame_size=(100, 64))
    p = DensityPredictor(model, frame_size=SIZE, max_batch=2)
    with pytest.raises(ValueError, match="outside"):
        p.run(3)
    with pytest.raises(ValueError, match="no masks"):
        p.run(1, mask_ids=[0])
    rgb_model = DensityNet().eval()
    with pytest.raises(ValueError, match="bgr"):
        DensityPredictor(rgb_model, frame_size=SIZE)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_fp16_matches_cpu_fp32(model, frames, masks) -> None:
    cpu = DensityPredictor(model, frame_size=SIZE, masks=masks, max_batch=8).predict(frames, mask_ids=[0] * 5)
    gpu = DensityPredictor(copy.deepcopy(model).to("cuda", torch.float16), frame_size=SIZE, masks=masks, max_batch=8)
    gpu.warmup()
    out = gpu.predict(frames, mask_ids=[0] * 5)
    np.testing.assert_allclose(out.counts, cpu.counts, rtol=2e-2)
