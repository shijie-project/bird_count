"""Image and points go through the same matrix, so they must stay aligned for
every combination of scale, crop/pad, mirror and rotation."""

import numpy as np
import pytest

from birdcount.training.data.geometry import crop_transform, warp_image, warp_points
from birdcount.training.data.targets import count_map


W, H, CROP = 320, 180, 128


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """Dark frame with bright 7x7 blobs centred on known points."""
    img = np.full((H, W, 3), 20, np.uint8)
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.integers(10, W - 10, 40), rng.integers(10, H - 10, 40)]).astype(np.float64) + 0.5
    for x, y in pts.astype(int):
        img[y - 3 : y + 4, x - 3 : x + 4] = 255
    return img, pts


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.3])
def test_points_land_on_their_blobs(seed, scale) -> None:
    img, pts = _scene()
    m = crop_transform(np.random.default_rng(seed), (W, H), CROP, scale)
    crop = warp_image(img, m, CROP)
    moved = warp_points(pts, m, CROP)
    assert crop.shape == (CROP, CROP, 3)
    for x, y in moved:
        assert crop[int(y), int(x)].min() > 150, (seed, x, y)


def test_padding_is_black() -> None:
    img = np.full((H, W, 3), 200, np.uint8)
    m = crop_transform(np.random.default_rng(0), (W, H), 512, 1.0, hflip_p=0, rot90=False)
    crop = warp_image(img, m, 512)
    assert (crop.reshape(-1, 3).min(axis=1) == 0).mean() == pytest.approx(1 - W * H / 512**2, abs=0.01)


def test_points_outside_the_crop_are_dropped() -> None:
    pts = np.array([[5.0, 5.0], [300.0, 170.0]])
    m = np.eye(3)
    assert len(warp_points(pts, m, CROP)) == 1


def test_mirror_and_rotation_are_exact_on_integers() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (CROP, CROP, 3), dtype=np.uint8)
    flip = np.array([[-1.0, 0, CROP], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(warp_image(img, flip, CROP), img[:, ::-1])
    rot = np.array([[0.0, 1, 0], [-1, 0, CROP], [0, 0, 1]])
    np.testing.assert_array_equal(warp_image(img, rot, CROP), np.rot90(img))


def test_count_map_mass_and_cells() -> None:
    pts = np.array([[0.0, 0.0], [7.99, 7.99], [8.0, 0.0], [15.5, 31.9], [15.5, 31.9]])
    m = count_map(pts, 32, 32, 8)
    assert m.sum() == len(pts)
    assert m[0, 0] == 2 and m[0, 1] == 1 and m[3, 1] == 2


def test_count_map_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        count_map(np.empty((0, 2)), 30, 32, 8)
