import cv2
import numpy as np
import pytest

from birdcount.core.visualize import (
    HEATMAP_MIN_DENSITY,
    HEATMAP_VMAX,
    MASK_EDGE_COLOR,
    Scratch,
    density_to_heatmap,
    mask_layers,
    overlay_density,
    overlay_mask,
)


def test_colour_scale_is_absolute() -> None:
    """Same density, same colour, whatever else is in the frame."""
    lone = np.zeros((4, 4), np.float32)
    lone[1, 1] = HEATMAP_VMAX / 2
    crowded = np.full((4, 4), HEATMAP_VMAX * 3, np.float32)
    crowded[1, 1] = HEATMAP_VMAX / 2
    heat_a, _ = density_to_heatmap(lone)
    heat_b, _ = density_to_heatmap(crowded)
    np.testing.assert_array_equal(heat_a[1, 1], heat_b[1, 1])


def test_saturates_at_vmax_and_masks_below_min_density() -> None:
    d = np.array([[0.0, HEATMAP_MIN_DENSITY * 0.5, HEATMAP_VMAX, HEATMAP_VMAX * 10]], np.float32)
    heat, mask = density_to_heatmap(d)
    assert mask.tolist() == [[0, 0, 255, 255]]
    np.testing.assert_array_equal(heat[0, 2], heat[0, 3])  # clipped
    red = cv2.applyColorMap(np.array([[255]], np.uint8), cv2.COLORMAP_JET)[0, 0]
    np.testing.assert_array_equal(heat[0, 3], red)


def test_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        density_to_heatmap(np.zeros((2, 3, 4)))


@pytest.mark.parametrize("smooth", [True, False])
def test_overlay_only_touches_visible_density(smooth) -> None:
    frame = np.full((64, 96, 3), 100, np.uint8)
    density = np.zeros((8, 12), np.float32)
    density[2:4, 3:5] = HEATMAP_VMAX
    out = overlay_density(frame.copy(), density, smooth=smooth)
    changed = (out != 100).any(axis=2)
    assert changed[16:32, 24:40].all()  # the hot cells, upsampled 8x
    assert not changed[:8].any() and not changed[:, 60:].any()


def test_overlay_of_empty_density_is_a_no_op() -> None:
    frame = np.full((64, 96, 3), 100, np.uint8)
    assert (overlay_density(frame.copy(), np.zeros((8, 12), np.float32)) == 100).all()


def test_mask_overlay_tints_ignored_area_and_outlines_boundary() -> None:
    keep = np.zeros((64, 96), np.uint8)
    keep[:, :48] = 1
    frame = np.full((64, 96, 3), 200, np.uint8)
    out = overlay_mask(frame.copy(), mask_layers(keep, (96, 64)))
    assert (out[:, :40] == 200).all()  # kept area untouched
    assert (out[:, 60:] != 200).any(axis=2).all()  # ignored area tinted
    assert (out[:, 47] == MASK_EDGE_COLOR).all()  # boundary drawn


def test_mask_layers_resize_to_the_target() -> None:
    keep = np.zeros((720, 1280), np.uint8)
    keep[:, :640] = 1
    ignored, edge = mask_layers(keep, (320, 180))
    assert ignored.shape == edge.shape == (180, 320)
    assert ignored[:, :150].sum() == 0 and ignored[:, 170:].all()


@pytest.mark.parametrize("smooth", [True, False])
def test_reused_scratch_draws_exactly_what_a_one_off_call_draws(smooth) -> None:
    rng = np.random.default_rng(0)
    keep = np.zeros((64, 96), np.uint8)
    keep[:, :48] = 1
    layers = mask_layers(keep, (96, 64))
    scratch = Scratch()
    for _ in range(3):  # same buffers, different frames
        frame = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        density = (rng.random((8, 12)) * HEATMAP_VMAX).astype(np.float32)
        one_off = overlay_mask(overlay_density(frame.copy(), density, smooth=smooth), layers)
        reused = overlay_mask(
            overlay_density(frame.copy(), density, smooth=smooth, scratch=scratch), layers, scratch=scratch
        )
        np.testing.assert_array_equal(reused, one_off)
