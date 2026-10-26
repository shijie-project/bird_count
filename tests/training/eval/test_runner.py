import json

import cv2
import numpy as np
import torch

from birdcount.core.inference import DensityPredictor
from birdcount.core.model import DensityNet
from birdcount.training.data.annotations import load_split
from birdcount.training.data.cache import FrameCache
from birdcount.training.eval.runner import evaluate, masks_for


MAC = "B8A44FD51C3C"


def test_masked_images_are_scored_on_their_region(tmp_path) -> None:
    data, masks_dir = tmp_path / "data", tmp_path / "masks"
    (data / "images" / "val").mkdir(parents=True)
    (data / "annotations").mkdir()
    masks_dir.mkdir()
    img = np.full((64, 96, 3), 120, np.uint8)
    for name in (f"cam_{MAC}_001", "no_mac_002"):
        cv2.imwrite(str(data / "images" / "val" / f"{name}.png"), img)
    mask = np.zeros((64, 96, 3), np.uint8)
    mask[:, :48] = 255  # left half counted
    cv2.imwrite(str(masks_dir / f"{MAC}.png"), mask)
    points = [[10.0, 10.0], [20.0, 30.0], [70.0, 30.0]]  # last one in the ignored half
    anns = [{"image_id": n, "points": points} for n in (f"cam_{MAC}_001", "no_mac_002")]
    (data / "annotations" / "val.json").write_text(json.dumps({"annotations": anns}))

    samples = load_split(data, "val")
    masks, ids = masks_for(samples, masks_dir, (96, 64))
    model = DensityNet().eval()
    model.set_input_order("bgr")
    predictor = DensityPredictor(model.requires_grad_(False), frame_size=(96, 64), masks=masks, max_batch=4)
    results = {
        r.name: r for r in evaluate(samples, predictor, FrameCache(tmp_path / "c", (96, 64)), ids, num_workers=0)
    }

    masked, plain = results[f"cam_{MAC}_001"], results["no_mac_002"]
    assert masked.masked and (masked.gt, masked.gt_outside_mask) == (2.0, 1)
    assert not plain.masked and (plain.gt, plain.gt_outside_mask) == (3.0, 0)
    assert (masked.density[:, 6:] == 0).all()  # density masked like deployment
    torch.testing.assert_close(
        torch.tensor(masked.pred), torch.tensor(float(masked.density.sum())), rtol=1e-3, atol=1e-4
    )
