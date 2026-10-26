import json

import cv2
import numpy as np
import pytest

from birdcount.training.labeling import cleanup


def write(path, image) -> None:
    assert cv2.imwrite(str(path), image)


def test_points_in_solid_black_are_dropped_dark_single_pixels_are_not(tmp_path) -> None:
    image = np.full((40, 60, 3), 120, np.uint8)
    image[:, :20] = 0  # masked-out band
    image[30, 45] = 0  # one dark pixel (a bird's eye, a shadow)
    write(tmp_path / "a.png", image)
    data = {
        "images": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "gone.png"}],
        "annotations": [
            {"image_id": "a", "points": [[5.0, 5.0], [17.0, 30.0], [45.0, 30.0], [50.0, 10.0]]},
            {"image_id": "gone", "points": [[1.0, 1.0]]},
        ],
    }
    r = cleanup.drop_masked_points(data, tmp_path)
    assert (r.before, r.after, r.per_image, r.missing) == (5, 3, [("a", 2, 2)], ["gone"])
    assert data["annotations"][0]["points"] == [[45.0, 30.0], [50.0, 10.0]]


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (72, 128, 3), dtype=np.uint8)


def test_duplicate_plan_keeps_the_labelled_copy(tmp_path) -> None:
    write(tmp_path / "a.png", _noise(0))
    write(tmp_path / "1a2b3c4d-a.png", _noise(0))  # LS upload of the same file: exact duplicate
    # The same picture saved again under a numbered name: other bytes, same pixels.
    cv2.imwrite(str(tmp_path / "b.png"), _noise(1), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cv2.imwrite(str(tmp_path / "017_b.png"), _noise(1), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    write(tmp_path / "017_c.png", _noise(2))
    write(tmp_path / "c.png", _noise(3))  # same name pattern, different picture

    plan = cleanup.duplicate_plan(tmp_path)
    assert [(r["duplicate"], r["keeper"], r["reason"]) for r in plan] == [
        ("017_b.png", "b.png", "near-reencoded"),
        ("1a2b3c4d-a.png", "a.png", "exact-sha256"),
    ]
    usage = cleanup.label_usage(
        [[{"data": {"img": "/data/upload/2/1a2b3c4d-a.png"}, "annotations": [{"result": [{}]}]}]]
    )
    plan = cleanup.duplicate_plan(tmp_path, usage)
    assert ("a.png", "1a2b3c4d-a.png") in [(r["duplicate"], r["keeper"]) for r in plan]

    run = cleanup.quarantine(tmp_path, plan, tmp_path / "_q")
    assert not (tmp_path / "a.png").exists() and (run / "a.png").is_file()
    assert [m["duplicate"] for m in json.loads((run / "manifest.json").read_text())["moves"]] == [
        r["duplicate"] for r in plan
    ]
    with pytest.raises(ValueError, match="inside"):
        cleanup.quarantine(tmp_path, plan, tmp_path.parent / "elsewhere")


def test_apply_masks_paints_the_ignored_region_black(tmp_path) -> None:
    masks, src, out = tmp_path / "masks", tmp_path / "src", tmp_path / "out"
    masks.mkdir(), src.mkdir()
    mask = np.zeros((36, 64, 3), np.uint8)  # drawn at another resolution than the images
    mask[:, :32] = 255
    write(masks / "B8A44FD51C3C_mask.png", mask)
    image = np.full((72, 128, 3), 200, np.uint8)
    write(src / "axis1_B8A44FD51C3C_0001.png", image)
    write(src / "axis1_B8A44FD52B7F_0002.png", image)  # no mask for this camera
    write(src / "nomac.png", image)

    r = cleanup.apply_masks(sorted(src.iterdir()), out, mask_dir=masks)
    assert r.written == ["axis1_B8A44FD51C3C_0001.png"]
    assert r.unmasked == ["axis1_B8A44FD52B7F_0002.png", "nomac.png"]
    result = cv2.imread(str(out / "axis1_B8A44FD51C3C_0001.png"))
    assert (result[:, :64] == 200).all() and (result[:, 64:] == 0).all()

    r = cleanup.apply_masks([src / "nomac.png"], out, mask_dir=None, mask=masks / "B8A44FD51C3C_mask.png")
    assert r.written == ["nomac.png"]
    with pytest.raises(ValueError, match="overwritten"):
        cleanup.apply_masks([src / "nomac.png"], src, mask_dir=masks)
