"""Compression-robust training: pair windows and settings, distillation crops, the held-out report, the trainer."""

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from birdcount.core.model import OUTPUT_STRIDE, DensityNet
from birdcount.core.settings import CoreSettings
from birdcount.training.distill.dataset import DistillCrops, Draws, holdout_report, read_index
from birdcount.training.distill.pairs import (
    ENCODERS,
    INDEX_FIELDS,
    StreamSetting,
    black_outside,
    camera_days,
    encoder_args,
    random_setting,
    windows_from_replay,
)
from birdcount.training.train.config import TrainConfig
from birdcount.training.train.trainer import Trainer

from ..data.conftest import make_split


MAC = "B8A44FD52E34"


def write_pairs(root: Path, frames: list[np.ndarray], densities: list[np.ndarray], settings=None) -> None:
    (root / "pairs").mkdir(parents=True)
    with open(root / "index.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        wr.writeheader()
        for i, (f, d) in enumerate(zip(frames, densities, strict=True)):
            cv2.imwrite(str(root / "pairs" / f"p{i}.png"), f)
            np.save(root / "pairs" / f"p{i}.npy", d.astype(np.float16))
            s = (settings or ["nv_p1ll 3fps 400k g60s"])[i % len(settings or [1])]
            wr.writerow({"file": f"p{i}", "mac": MAC, "setting": s, "teacher": float(d.sum()),
                         "teacher_on_delivered": float(d.sum()) * 0.9, "dense": 1})  # fmt: skip


def test_stream_settings_are_complete_and_encode_their_cap() -> None:
    rng = random.Random(0)
    drawn = [random_setting(rng) for _ in range(300)]
    assert {s.enc for s in drawn} == {"clean", *ENCODERS}
    for s in drawn:
        if s.enc != "clean":
            args = encoder_args(s)
            assert f"{s.kbps}k" in args and str(s.gop * s.fps) in args and args[args.index("-bf") + 1] == "0"
    with pytest.raises(ValueError):
        StreamSetting("nv_p1ll", 3)  # no bitrate, no GOP
    assert StreamSetting("x264_med", 5, 400, 60, masked=True).key == "x264_med 5fps 400k g60s masked"


def test_black_outside_keeps_the_region_and_blacks_the_rest() -> None:
    h, w = 4, 8
    keep = np.zeros((h, w), np.uint8)
    keep[:, :4] = 1
    frame = np.full(w * h * 3 // 2, 200, np.uint8)
    out = black_outside([frame], keep)[0]
    y = out[: w * h].reshape(h, w)
    assert (y[:, :4] == 200).all() and (y[:, 4:] == 16).all()
    u = out[w * h : w * h * 5 // 4].reshape(h // 2, w // 2)
    assert (u[:, :2] == 200).all() and (u[:, 2:] == 128).all()


def test_camera_days_come_from_names_and_paths() -> None:
    days = camera_days(
        [Path(f"axis1/axis-{MAC}/20260325_163554_BE6C/20260325_163554_BE6C_original.mkv"), Path("notes.txt")]
    )
    assert days == {(MAC, "2026-03-25")}


def test_replay_windows_prefer_crowded_stretches_and_skip_excluded_days(tmp_path) -> None:
    cam = tmp_path / MAC
    cam.mkdir()
    t = np.arange(0, 600, 0.2)  # 10 min at 5 fps, two clips of 5 min
    count = np.where(t < 300, 20.0, 150.0)
    clip = (t >= 300).astype(int)
    np.savez(cam / "2026-03-11.npz", t=t, count=count, clip=clip, clips=np.array(["a.mkv", "b.mkv"], dtype=object))
    np.savez(cam / "2026-03-12.npz", t=t, count=count, clip=clip, clips=np.array(["c.mkv", "d.mkv"], dtype=object))
    windows = windows_from_replay(
        tmp_path, {MAC: 160.0}, exclude={(MAC, "2026-03-12")}, rng=random.Random(0), exists=lambda p: True
    )
    assert {w.day for w in windows} == {"2026-03-11"}
    dense = windows[:3]
    assert all(w.clip.name == "b.mkv" and 0 <= w.offset <= 240 and w.offset % 60 == 0 for w in dense)
    assert len(windows) == 4


def test_distill_crops_move_image_and_density_together(tmp_path) -> None:
    h = w = 64
    frame = np.zeros((h, w, 3), np.uint8)
    frame[8:16, 40:48] = 255  # one bright cell: row 1, column 5 at stride 8
    density = np.zeros((h // OUTPUT_STRIDE, w // OUTPUT_STRIDE), np.float32)
    density[1, 5] = 3.0
    write_pairs(tmp_path, [frame], [density])
    ds = DistillCrops(tmp_path, crop=64)
    for seed in range(8):
        item = ds[(0, seed)]
        img, den = item["image"].numpy(), item["density"][0].numpy()
        assert img.shape == (64, 64, 3) and den.shape == (8, 8) and den.sum() == pytest.approx(3.0)
        r, c = np.unravel_index(den.argmax(), den.shape)
        assert img[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8].min() == 255  # the mass sits on the bright cell


def test_draws_are_endless_and_reproducible() -> None:
    a = [d for _, d in zip(range(50), Draws(5, seed=1), strict=False)]
    b = [d for _, d in zip(range(50), Draws(5, seed=1), strict=False)]
    assert a == b and {i for i, _ in a} == set(range(5))


def test_holdout_report_measures_bias_on_crowded_frames() -> None:
    rows = [
        {"setting": "s", "teacher": "100", "teacher_on_delivered": "90", "dense": "1"},
        {"setting": "s", "teacher": "100", "teacher_on_delivered": "94", "dense": "1"},
        {"setting": "s", "teacher": "10", "teacher_on_delivered": "10", "dense": "0"},
    ]
    r = holdout_report(rows, np.array([99.0, 99.0, 12.0]))["s"]
    assert r["dense_n"] == 2 and r["dense_bias_pct"] == -1.0 and r["dense_bias_pct_before"] == -8.0
    assert r["mae"] == pytest.approx(4 / 3, abs=1e-3)


@pytest.mark.slow
def test_training_with_distillation_logs_its_losses_and_the_holdout(tmp_path) -> None:
    W, H = 128, 96
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    images = {f"img{i}": (rng.integers(0, 256, (H, W, 3), dtype=np.uint8), (rng.random((5, 2)) * [W, H]).tolist())
              for i in range(4)}  # fmt: skip
    make_split(data, "train", images)
    make_split(data, "val", dict(list(images.items())[:2]))
    frames = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(6)]
    dens = [rng.random((H // 8, W // 8)).astype(np.float32) * 0.1 for _ in range(6)]
    write_pairs(tmp_path / "pairs", frames, dens)
    write_pairs(tmp_path / "holdout", frames[:4], dens[:4], ["a 3fps", "clean 3fps"])
    init = tmp_path / "init.pth"
    torch.manual_seed(0)
    torch.save(DensityNet().state_dict(), init)
    settings = CoreSettings(
        data_root=data, output_root=tmp_path / "out", mask_dir=None, image_width=W, image_height=H, log_level="WARNING"
    )
    config = TrainConfig(
        output_dir=tmp_path / "run", crop=64, train_sizes=(128,), batch_size=2, num_workers=0, epochs=2,
        warmup_epochs=1, val_start=0, val_every=1, device="cpu", init=init, ot_iterations=10,
        distill=tmp_path / "pairs", distill_holdout=tmp_path / "holdout", distill_weight=0.5,
    )  # fmt: skip
    trainer = Trainer(config, settings)
    trainer.setup()
    summary = trainer.run()
    assert summary["status"] == "finished"
    records = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(np.isfinite(r["train"]["distill_count"]) and r["train"]["distill_map"] > 0 for r in records)
    assert set(records[-1]["val"]["holdout"]) == {"a 3fps", "clean 3fps"}
    assert len(read_index(tmp_path / "holdout")) == 4
