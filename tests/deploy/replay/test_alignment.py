import cv2
import numpy as np

from birdcount.deploy.replay.alignment import register


def fittings(seed: int = 0) -> np.ndarray:
    """A shed's fixed fittings, as far as feature matching cares: edges and corners at random."""
    rng = np.random.default_rng(seed)
    img = np.full((720, 1280, 3), 90, np.uint8)
    for _ in range(160):
        x, y = int(rng.integers(0, 1200)), int(rng.integers(0, 660))
        w, h = int(rng.integers(10, 80)), int(rng.integers(10, 60))
        cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def test_a_camera_that_did_not_move_fits() -> None:
    ref = fittings()
    fit = register(ref, ref.copy())
    assert fit.verdict(6.0) == "ok" and fit.shift < 0.5


def test_a_camera_moved_by_a_few_degrees_and_pixels_is_caught() -> None:
    ref = fittings()
    moved = cv2.getRotationMatrix2D((640, 360), 1.5, 1.0)
    moved[:, 2] += (25, -12)  # and shifted
    turned = cv2.warpAffine(ref, moved, (1280, 720))
    fit = register(ref, turned)
    assert fit.verdict(6.0) == "MOVED"
    assert abs(fit.angle + 1.5) < 0.2  # OpenCV turns counter-clockwise for a positive angle; y points down
    assert abs(fit.scale - 1) < 0.01


def test_pictures_with_nothing_in_common_are_not_called_a_fit() -> None:
    fit = register(fittings(0), fittings(1))
    assert fit is None or fit.verdict(6.0) != "ok"
