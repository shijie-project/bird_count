"""The trainer end to end on a tiny synthetic dataset, CPU: run, interrupt, resume."""

import json

import numpy as np
import pytest
import torch

from birdcount.core.model import DensityNet
from birdcount.core.settings import CoreSettings
from birdcount.training.train.config import TrainConfig
from birdcount.training.train.trainer import Trainer

from ..data.conftest import make_split


W, H = 128, 96


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)

    def images(n):
        return {
            f"img{i}": (rng.integers(0, 256, (H, W, 3), dtype=np.uint8), (rng.random((5 + i, 2)) * [W, H]).tolist())
            for i in range(n)
        }

    data = tmp_path / "data"
    make_split(data, "train", images(6))
    make_split(data, "val", images(3))
    init = tmp_path / "init.pth"
    torch.manual_seed(0)
    torch.save(DensityNet().state_dict(), init)
    settings = CoreSettings(
        data_root=data, output_root=tmp_path / "out", mask_dir=None, image_width=W, image_height=H, log_level="WARNING"
    )
    config = TrainConfig(
        output_dir=tmp_path / "run",
        crop=64,
        train_sizes=(128,),
        batch_size=4,
        num_workers=0,
        epochs=4,
        warmup_epochs=1,
        val_start=0,
        val_every=1,
        device="cpu",
        init=init,
        hard_mining_gamma=1.0,
        hard_mining_start=1,
        ot_iterations=20,
    )
    return config, settings


def lines(run):
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.slow
def test_a_run_writes_everything_and_resumes_where_it_stopped(setup, monkeypatch) -> None:
    config, settings = setup
    run = config.output_dir

    trainer = Trainer(config, settings)
    trainer.setup()
    real = Trainer.train_epoch

    def stop_at_two(self, epoch):
        if epoch == 2:
            raise KeyboardInterrupt
        return real(self, epoch)

    monkeypatch.setattr(Trainer, "train_epoch", stop_at_two)
    summary = trainer.run()
    assert summary["status"] == "interrupted" and summary["epochs_completed"] == 2
    assert [r["epoch"] for r in lines(run)] == [0, 1]
    assert torch.load(run / "last.tar", weights_only=True)["epoch"] == 1

    monkeypatch.setattr(Trainer, "train_epoch", real)
    resumed = Trainer(TrainConfig(resume=run), settings)
    resumed.setup()
    assert resumed.start_epoch == 2 and resumed.config == TrainConfig.load(run / "config.json")
    summary = resumed.run()
    assert summary["status"] == "finished" and summary["epochs_completed"] == 4

    records = lines(run)
    assert [r["epoch"] for r in records] == [0, 1, 2, 3]
    assert all("val" in r and np.isfinite(r["train"]["loss"]) for r in records)
    best = summary["best"]
    assert best is not None and (run / best["checkpoint"].split("\\")[-1].split("/")[-1]).is_file()
    assert {"mae", "rmse", "counting_accuracy", "primary_accuracy"} <= set(best)
    saved = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "finished" and saved["config"]["crop"] == 64
    # The best file is what deployment loads.
    from birdcount.core.model import load_inference_model

    load_inference_model(best["checkpoint"], "cpu")
