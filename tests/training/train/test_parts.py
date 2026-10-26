"""Loss, EMA, config, checkpoints, sweep: the trainer's pieces on their own."""

import json

import pytest
import torch
from torch import nn

from birdcount.training.train import sweep
from birdcount.training.train.checkpoints import BestCheckpoints, best_name, save_atomic
from birdcount.training.train.config import TrainConfig, build_parser, config_from_args
from birdcount.training.train.ema import ModelEMA
from birdcount.training.train.losses import DMCountLoss


# --- loss ---------------------------------------------------------------------


def test_loss_parts_add_up_and_aux_can_be_switched_off() -> None:
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 8, 8) * 0.1
    target = torch.zeros(2, 1, 8, 8)
    target[0, 0, 2, 3] = 2
    target[1, 0, 5, 5] = 1
    points = [torch.tensor([[25.0, 18.0], [26.0, 19.0]]), torch.tensor([[44.0, 44.0]])]
    loss = DMCountLoss(64, 8, norm_coords=True)
    total, parts = loss(pred, target, points)
    assert total.item() == pytest.approx(sum(parts[k].item() for k in ("ot", "count", "tv", "aux")), rel=1e-5)
    assert parts["count"].item() == pytest.approx(
        (pred.flatten(1).sum(1) - torch.tensor([2.0, 1.0])).abs().mean().item()
    )
    _, no_aux = DMCountLoss(64, 8, norm_coords=True, aux_sigma=0)(pred, target, points)
    assert no_aux["aux"].item() == 0


# --- EMA ----------------------------------------------------------------------


def test_ema_blends_floats_and_copies_counters() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = ModelEMA(model, decay=0.75)
    before = ema.model[0].weight.clone()
    with torch.no_grad():
        model[0].weight.add_(1.0)
    model[1].num_batches_tracked += 3
    ema.update(model)
    torch.testing.assert_close(ema.model[0].weight, 0.75 * before + 0.25 * model[0].weight)
    assert ema.model[1].num_batches_tracked.item() == 3
    assert not any(p.requires_grad for p in ema.model.parameters())


# --- config -------------------------------------------------------------------


def test_command_line_and_json_round_trip() -> None:
    args = build_parser().parse_args(["--no-ema", "--lr", "5e-5", "--train-sizes", "1024", "1280", "--init", "w.pth"])
    cfg = config_from_args(args)
    assert (cfg.ema, cfg.lr, cfg.train_sizes, str(cfg.init)) == (False, 5e-5, (1024, 1280), "w.pth")
    assert TrainConfig.from_json(json.loads(json.dumps(cfg.to_json()))) == cfg
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_json({"lr": 1.0, "nope": 2})
    with pytest.raises(ValueError, match="multiple of 8"):
        TrainConfig(crop=500)


def test_defaults_are_the_production_recipe() -> None:
    cfg = TrainConfig()
    assert (cfg.lr, cfg.epochs, cfg.ema, cfg.ema_decay, cfg.norm_coords, cfg.wot, cfg.waux) == (
        1e-5,
        1000,
        True,
        0.996,
        True,
        0.1,
        1.0,
    )


# --- checkpoints --------------------------------------------------------------


def test_best_checkpoints_rotate_and_survive_a_resume(tmp_path) -> None:
    best = BestCheckpoints(tmp_path, keep=2)
    for epoch in (5, 10, 15):
        best.add({"w": torch.tensor([epoch])}, epoch, 10.0 - epoch / 10, 12.0)
    assert sorted(p.name for p in tmp_path.glob("best_*.pth")) == [best_name(10, 9.0, 12.0), best_name(15, 8.5, 12.0)]
    again = BestCheckpoints(tmp_path, keep=2)  # a resumed run finds them
    again.add({"w": torch.tensor([20])}, 20, 8.0, 11.0)
    assert sorted(p.name for p in tmp_path.glob("best_*.pth")) == [best_name(15, 8.5, 12.0), best_name(20, 8.0, 11.0)]
    save_atomic({"x": 1}, tmp_path / "last.tar")
    assert not list(tmp_path.glob("*.tmp"))


# --- sweep --------------------------------------------------------------------


def test_sweep_grid_and_flags() -> None:
    grid = sweep.parse_grid(["lr=1e-5,5e-5", "--ema=true,false"])
    combos = sweep.combinations(grid)
    assert len(combos) == 4 and sweep.run_name(combos[1]) == "lr=1e-5,ema=false"
    assert sweep.flags(combos[1]) == ["--lr", "1e-5", "--no-ema"]
    with pytest.raises(ValueError, match="not a training option"):
        sweep.parse_grid(["lrate=1"])
    with pytest.raises(ValueError, match="key=v1"):
        sweep.parse_grid(["lr"])


def test_leaderboard_ranks_by_the_checkpoint_score(tmp_path) -> None:
    for name, mae, rmse in (("a", 4.0, 6.0), ("b", 5.0, 5.0), ("c", None, None)):
        (tmp_path / name).mkdir()
        best = (
            {"epoch": 1, "mae": mae, "rmse": rmse, "counting_accuracy": 0.9, "primary_accuracy": 0.8} if mae else None
        )
        (tmp_path / name / "summary.json").write_text(json.dumps({"status": "finished", "best": best}))
    assert [r["run"] for r in sweep.leaderboard(tmp_path)] == ["b", "a", "c"]  # 15 < 16; no validation last
