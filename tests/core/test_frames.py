import cv2
import numpy as np
import pytest

from birdcount.core.frames import fit_frame, read_image


def test_fit_frame_is_free_at_model_size() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert fit_frame(frame) is frame


def test_fit_frame_squashes_without_keeping_aspect() -> None:
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1440, 3), dtype=np.uint8)
    out = fit_frame(frame, (1280, 720))
    assert out.shape == (720, 1280, 3)
    np.testing.assert_array_equal(out, cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR))


def test_read_image_handles_non_ascii_paths(tmp_path) -> None:
    folder = tmp_path / "鸡舍 masks"
    folder.mkdir()
    img = np.full((8, 16, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(folder / "掩码.png")
    np.testing.assert_array_equal(read_image(folder / "掩码.png"), img)


def test_read_image_fails_loudly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    (tmp_path / "junk.png").write_bytes(b"not an image")
    with pytest.raises(ValueError):
        read_image(tmp_path / "junk.png")
