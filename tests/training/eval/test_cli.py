"""End to end on a tiny synthetic split, CPU, random weights."""

import csv
import json

import cv2
import numpy as np
import torch

from birdcount.core.model import DensityNet
from birdcount.training.eval.__main__ import main


def test_eval_writes_every_artifact(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    (data / "images" / "val").mkdir(parents=True)
    (data / "annotations").mkdir()
    rng = np.random.default_rng(0)
    anns = []
    for i in range(3):
        cv2.imwrite(str(data / "images" / "val" / f"img{i}.png"), rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
        anns.append({"image_id": f"img{i}", "points": [[10.0 * (i + 1), 20.0]] * (i + 1)})
    (data / "annotations" / "val.json").write_text(json.dumps({"annotations": anns}))
    ckpt = tmp_path / "m.pth"
    torch.manual_seed(0)
    torch.save(DensityNet().state_dict(), ckpt)

    for key, value in {"DATA_ROOT": data, "OUTPUT_ROOT": tmp_path / "out", "MASK_DIR": ""}.items():
        monkeypatch.setenv(key, str(value))
    out = tmp_path / "run"
    argv = [
        "--split",
        "val",
        "--ckpt",
        str(ckpt),
        "--device",
        "cpu",
        "--frame-size",
        "96x64",
        "--workers",
        "0",
        "--blobs",
        "--grid",
        "2x3",
        "--output-dir",
        str(out),
    ]
    assert main(argv) == 0

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["run"]["images"] == 3 and metrics["overall"]["total_gt"] == 6
    assert {"blobs", "grid", "tolerance_accuracy", "pileup_detection"} <= set(metrics)
    with (out / "per_image.csv").open() as f:
        assert [r["name"] for r in csv.DictReader(f)] == ["img0", "img1", "img2"]
    assert len(json.loads((out / "grid.json").read_text())["images"][0]["regions"]) == 6
    assert sorted(p.name for p in (out / "overlays").iterdir())[:3] == [
        "img0_blobs.png",
        "img0_counts.png",
        "img0_grid.png",
    ]
