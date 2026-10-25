"""Region-mask loading. The numbers that matter: a mask keeps exactly the
white area, unmatched streams are a genuine no-op, and a density cell
straddling the boundary is weighted by its overlap (the reason the downsample
is INTER_AREA)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from birdcount.core.masks import RegionMasks, density_weights, read_keep_mask


MAC_A = "B8A44FD51C3C"
MAC_B = "B8A44F5B4394"
W, H = 96, 64
SIZE = (W, H)


def _write_mask(path: Path, keep_left_fraction: float, scale: int = 1) -> Path:
    """White (count) on the left, black (ignore) on the right."""
    img = np.zeros((H * scale, W * scale, 3), np.uint8)
    img[:, : int(W * scale * keep_left_fraction)] = 255
    cv2.imwrite(str(path), img)
    return path


def test_keep_mask_follows_the_black_level(tmp_path) -> None:
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :48] = 255
    img[:, 48:60] = (9, 9, 9)  # black: every channel < 10
    img[:, 60:] = (0, 0, 10)  # not black: one channel reaches 10
    cv2.imwrite(str(tmp_path / "m.png"), img)
    keep = read_keep_mask(tmp_path / "m.png", SIZE)
    assert keep[:, :48].all() and not keep[:, 48:60].any() and keep[:, 60:].all()


def test_oversized_mask_is_resized_nearest(tmp_path) -> None:
    keep = read_keep_mask(_write_mask(tmp_path / "m.png", 0.5, scale=3), SIZE)
    assert keep.shape == (H, W)
    assert set(np.unique(keep)) == {0, 1}
    assert keep.mean() == pytest.approx(0.5, abs=0.01)


def test_fully_black_mask_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="fully black"):
        read_keep_mask(_write_mask(tmp_path / "m.png", 0.0), SIZE)


def test_from_dir_matches_bare_and_decorated_names(tmp_path) -> None:
    _write_mask(tmp_path / f"{MAC_A}.png", 0.5)
    _write_mask(tmp_path / f"mask_axis2_{MAC_B}.png", 0.25)
    masks = RegionMasks.from_dir(tmp_path, [MAC_A, MAC_B], SIZE)
    assert masks.covered == {0: MAC_A, 1: MAC_B}
    assert masks.uncovered == []
    assert masks.keep[0].mean() == pytest.approx(0.5, abs=0.01)
    assert masks.keep[1].mean() == pytest.approx(0.25, abs=0.01)


def test_unmatched_stream_gets_an_all_ones_row(tmp_path) -> None:
    _write_mask(tmp_path / f"{MAC_A}.png", 0.5)
    masks = RegionMasks.from_dir(tmp_path, [MAC_A, None, MAC_B], SIZE)
    assert masks.covered == {0: MAC_A}
    assert masks.uncovered == [1, 2]
    assert masks.keep[1].min() == 1 and masks.keep[2].min() == 1


@pytest.mark.parametrize("mask_dir", [None, "", "does/not/exist"])
def test_from_dir_off_when_unset_or_missing(mask_dir) -> None:
    assert RegionMasks.from_dir(mask_dir, [MAC_A], SIZE) is None


def test_ambiguous_and_black_masks_degrade_to_unmasked(tmp_path) -> None:
    _write_mask(tmp_path / f"old_{MAC_A}.png", 0.5)
    _write_mask(tmp_path / f"new_{MAC_A}.png", 0.9)
    _write_mask(tmp_path / f"{MAC_B}.png", 0.0)
    assert RegionMasks.from_dir(tmp_path, [MAC_A, MAC_B], SIZE) is None


def test_for_source_finds_the_mask_by_the_videos_mac(tmp_path) -> None:
    _write_mask(tmp_path / f"{MAC_A}_mask.png", 0.5)
    masks = RegionMasks.for_source(tmp_path, f"/videos/{MAC_A}_20260314_070001_82E5.mkv", SIZE)
    assert masks.covered == {0: MAC_A}
    assert RegionMasks.for_source(tmp_path, "/videos/no_mac_here.mkv", SIZE) is None


def test_from_file_must_load(tmp_path) -> None:
    masks = RegionMasks.from_file(_write_mask(tmp_path / f"{MAC_A}.png", 0.5), SIZE)
    assert masks.covered == {0: MAC_A}
    with pytest.raises(FileNotFoundError):
        RegionMasks.from_file(tmp_path / "missing.png", SIZE)


def test_density_weights_drop_exactly_the_masked_area(tmp_path) -> None:
    keep = read_keep_mask(_write_mask(tmp_path / "m.png", 0.5), SIZE)
    weights = density_weights(keep, 8)
    assert weights.shape == (H // 8, W // 8)
    assert weights.mean() == pytest.approx(0.5, abs=0.01)


def test_boundary_cells_are_area_weighted_not_rounded(tmp_path) -> None:
    # A boundary at 7/16 of the width falls mid-cell after an 8x downsample;
    # NEAREST would round that column to 0 or 1 and lose a strip of birds.
    keep = read_keep_mask(_write_mask(tmp_path / "m.png", 0.4375), SIZE)
    weights = density_weights(keep[None], 8)[0]
    assert ((weights > 0.01) & (weights < 0.99)).any()
    assert weights.mean() == pytest.approx(0.4375, abs=0.01)
