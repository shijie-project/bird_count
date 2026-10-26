"""Run folders as the tools write them, and a tiny dataset."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_train(run: Path, epochs: int = 4, *, finished: bool = True, partial_line: bool = False) -> Path:
    write_json(run / "config.json", {"epochs": 10, "name": "x"})
    lines = []
    for e in range(epochs):
        record = {"epoch": e, "lr": 1e-5, "seconds": 0.5, "train": {"loss": 10.0 - e, "mae": 5.0 - e, "rmse": 6.0}}
        if e % 2:
            record["val"] = {"mae": 4.0 - e / 10, "rmse": 5.0, "counting_accuracy": 0.9, "primary_accuracy": 0.8}
        lines.append(json.dumps(record))
    text = "".join(line + "\n" for line in lines) + ('{"epoch": 9, "lr"' if partial_line else "")
    (run / "metrics.jsonl").write_text(text, encoding="utf-8")
    (run / "best_ep0003_mae3.70_rmse5.00.pth").write_bytes(b"x")
    (run / "train.log").write_bytes(b"epoch 0\nepoch 1\n")
    if finished:
        best = {"epoch": 3, "mae": 3.7, "rmse": 5.0, "counting_accuracy": 0.9, "primary_accuracy": 0.8}
        write_json(run / "summary.json", {"status": "finished", "best": best, "seconds": 12.0})
    return run


def make_eval(run: Path, split: str = "val") -> Path:
    images = [("a", 10, 12.0), ("b", 20, 15.0)]
    write_json(run / "metrics.json", {
        "run": {"split": split, "images": 2, "checkpoint": "m.pth"},
        "overall": {"counting_accuracy": 0.9, "mae": 3.5, "rmse": 4.0},
        "tolerance_accuracy": {"primary": {"accuracy": 0.5}},
    })  # fmt: skip
    rows = ["name,gt,pred,error,abs_error,rel_error_pct,within_primary_tol,masked,gt_outside_mask"]
    rows += [f"{n},{g},{p},{p - g:+.4f},{abs(p - g):.4f},{(p - g) / g * 100:+.4f},1,0,0" for n, g, p in images]
    (run / "per_image.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    regions = [{"id": 1, "gt": 5, "count": 7.0, "error": 2.0}, {"id": 2, "gt": 3, "count": 3.0, "error": 0.0}]
    write_json(run / "blobs.json", {"params": {}, "images": [{"name": "a", "regions": regions}]})
    (run / "overlays").mkdir()
    for name, _, _ in images:
        for kind in ("counts", "blobs"):
            (run / "overlays" / f"{name}_{kind}.png").write_bytes(b"png")
    return run


def make_video(run: Path, samples: int = 5, *, status: str = "finished") -> Path:
    record = {"status": status, "video": {"name": "v.mkv"}, "settings": {"window_px": 128}}
    if status != "running":
        stat = {"max": 9.0, "mean": 5.0, "p95": 8.0, "min": 1.0, "max_t": 1.0, "max_clock": "0:00:01"}
        record |= {"summary": {"samples": samples, "count": stat, "peak": stat}, "artifacts": {"peaks": []}}
    write_json(run / "summary.json", record)
    lines = [
        json.dumps({"i": i, "frame": i * 25, "t": float(i), "count": 5.0 + i, "peak": 1.0, "box": [0, 0, 8, 8]})
        for i in range(samples)
    ]
    (run / "samples.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return run


@pytest.fixture
def output_root(tmp_path) -> Path:
    root = tmp_path / "out"
    make_train(root / "train" / "20261001-100000-a")
    make_train(root / "train" / "20261002-100000-b", finished=False, partial_line=True)
    make_train(root / "train" / "sweep-20261003-100000" / "lr=1")
    make_eval(root / "eval" / "best" / "val-20261004-100000")
    make_video(root / "video" / "clip" / "20261005-100000")
    return root


@pytest.fixture
def data_root(tmp_path) -> Path:
    data = tmp_path / "data"
    (data / "images" / "val").mkdir(parents=True)
    for name in ("a", "b"):
        cv2.imwrite(str(data / "images" / "val" / f"{name}.jpg"), np.zeros((8, 8, 3), np.uint8))
    write_json(data / "annotations" / "val.json", {
        "images": [], "annotations": [{"image_id": "a", "points": [[1.0, 2.0], [3.0, 4.0]]}],
    })  # fmt: skip
    return data
