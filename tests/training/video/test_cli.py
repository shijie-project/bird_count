"""End to end on a tiny clip: CPU, random weights, region mask by MAC."""

import csv
import json

import cv2
import numpy as np
import torch

from birdcount.core.inference import DensityPredictor
from birdcount.core.masks import RegionMasks
from birdcount.core.model import DensityNet, load_inference_model
from birdcount.training.video.__main__ import main

from .conftest import MAC, SIZE


def test_one_pass_writes_every_artifact_and_counts_as_deployed(clip, tmp_path, monkeypatch) -> None:
    path, frames = clip
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    mask = np.zeros((64, 96, 3), np.uint8)
    mask[:, :64] = 255
    cv2.imwrite(str(masks_dir / f"{MAC}.png"), mask)
    ckpt = tmp_path / "m.pth"
    torch.manual_seed(0)
    torch.save(DensityNet().state_dict(), ckpt)
    env = {"MASK_DIR": masks_dir, "OUTPUT_ROOT": tmp_path / "out", "IMAGE_WIDTH": SIZE[0], "IMAGE_HEIGHT": SIZE[1]}
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    out = tmp_path / "run"
    argv = [str(path), "--ckpt", str(ckpt), "--device", "cpu", "--batch-size", "4", "--output-dir", str(out)]
    argv += ["--every", "0.5", "--frames", "boxes", "--render", "split", "--peaks", "2", "--peak-gap", "1"]
    argv += ["--count-threshold", "1"]
    assert main(argv) == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "finished"
    assert summary["settings"]["masked"] and summary["settings"]["step"] == 5
    assert summary["settings"]["start_time"] == "2026-03-12 10:30:23"
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [s["frame"] for s in samples] == list(range(0, 60, 5))
    with (out / "timeline.csv").open(encoding="utf-8") as f:
        assert [r["clock"] for r in csv.DictReader(f)][:2] == ["10:30:23", "10:30:23"]
    assert len(list((out / "frames").iterdir())) == len(samples)
    assert [p["path"] for p in summary["artifacts"]["peaks"]] == sorted(
        p["path"] for p in summary["artifacts"]["peaks"]
    )
    assert len(list((out / "peaks").iterdir())) == 2
    assert (out / "timeline.png").stat().st_size > 0
    video = cv2.VideoCapture(str(out / "split.mp4"))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == len(samples)
    assert (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), video.get(cv2.CAP_PROP_FPS)) == (2 * SIZE[0], 2.0)
    video.release()

    # The counts are DensityPredictor's on the same frames, masked by the camera's mask.
    model = load_inference_model(ckpt, "cpu", input_order="bgr")
    predictor = DensityPredictor(
        model, frame_size=SIZE, masks=RegionMasks.for_source(masks_dir, path, SIZE), max_batch=4
    )
    expected = predictor.predict([frames[s["frame"]] for s in samples], [0] * len(samples)).counts
    np.testing.assert_allclose([s["count"] for s in samples], expected, atol=1e-3)
