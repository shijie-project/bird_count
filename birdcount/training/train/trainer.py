"""The training loop.

Per step: uint8 crops from the workers -> colour augmentation and
normalisation on the GPU -> model -> DM-Count loss -> AdamW (+ EMA). Step
statistics stay on the GPU and come back once per epoch (v2 synced every
step), along with the per-crop errors hard-example mining needs.

Validation scores the EMA weights the way they will be deployed: copied into
a fused fp16 inference model (`core.model.prepare_inference_model`) and run
through `training.eval` — full frames at the deployment geometry, region
masks applied. The best score, 2·RMSE + MAE, picks the checkpoints kept.

Everything a run produces goes to its folder:

    config.json      the full TrainConfig (what --resume continues with)
    train.log        the human-readable log
    metrics.jsonl    one JSON object per epoch: train losses, and val metrics when validated
    summary.json     how the run ended and its best checkpoint, with all its metrics
    last.tar         resume state; best_ep*_mae*_rmse*.pth  the kept best weights
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from birdcount.core.inference import DensityPredictor
from birdcount.core.log import setup_logging
from birdcount.core.model import OUTPUT_STRIDE, DensityNet, build_model, prepare_inference_model
from birdcount.core.paths import PROJECT_ROOT
from birdcount.core.settings import CoreSettings
from birdcount.training import runs
from birdcount.training.data.annotations import load_split
from birdcount.training.data.cache import FrameCache
from birdcount.training.data.datasets import CropConfig, TrainCrops
from birdcount.training.data.loader import make_train_loader
from birdcount.training.data.photometric import Photometric
from birdcount.training.data.sampling import EpochSampler
from birdcount.training.eval.metrics import PRIMARY_ABS_TOL, PRIMARY_REL_TOL, compute_metrics, fraction_within
from birdcount.training.eval.report import write_json
from birdcount.training.eval.runner import evaluate, masks_for
from birdcount.training.train.checkpoints import BestCheckpoints, save_atomic
from birdcount.training.train.config import TrainConfig
from birdcount.training.train.ema import ModelEMA
from birdcount.training.train.hard_mining import HardExampleTracker
from birdcount.training.train.losses import DMCountLoss


logger = logging.getLogger(__name__)

PARTS = ("ot", "count", "tv", "aux", "wd")
VAL_BATCH = 8


class Trainer:
    def __init__(self, config: TrainConfig, settings: CoreSettings):
        self.settings = settings
        if config.resume is not None:
            run_dir = Path(config.resume)
            config = TrainConfig.load(run_dir / "config.json")
        else:
            run_dir = config.output_dir or runs.train_dir(settings.output_root, config.name)
        self.config = config
        self.run_dir = run_dir
        self.device = torch.device(config.device)
        self.start_epoch = 0
        self.best_score = math.inf
        self.evals_since_best = 0
        self.best: dict | None = None  # the best validation's metrics + checkpoint

    # ------------------------------------------------------------------
    # Set-up
    # ------------------------------------------------------------------

    def setup(self) -> None:
        cfg, s = self.config, self.settings
        resuming = (self.run_dir / "last.tar").is_file()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(s.log_level, log_file=self.run_dir / "train.log")
        if not resuming:
            cfg.save(self.run_dir / "config.json")
        logger.info("Run folder: %s", self.run_dir)

        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = not cfg.deterministic
        torch.backends.cudnn.deterministic = cfg.deterministic

        data_root = cfg.data_root or s.data_root
        cache = FrameCache(s.output_root / "cache", s.frame_size)
        train = load_split(data_root, "train")
        self.val_samples = load_split(data_root, "val")
        self.cache = cache
        self.dataset = TrainCrops(train, cache, CropConfig(crop=cfg.crop, train_sizes=cfg.train_sizes))
        self.sampler = EpochSampler(len(self.dataset), seed=cfg.seed)
        self.loader = make_train_loader(
            self.dataset, self.sampler, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
        self.photometric = Photometric()
        self.rng = torch.Generator().manual_seed(cfg.seed)  # photometric draws, reproducible across resumes
        self.val_masks, self.val_mask_ids = (
            masks_for(self.val_samples, s.mask_dir, s.frame_size) if cfg.masks else (None, None)
        )
        logger.info(
            "Data: %d train / %d val images from %s; %d steps per epoch.",
            len(train),
            len(self.val_samples),
            data_root,
            len(self.loader),
        )

        self.model = build_model(
            cfg.init, pretrained_backbone=cfg.init is None, freeze_backbone_bn=cfg.freeze_backbone_bn
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = self._scheduler()
        self.loss = DMCountLoss(
            cfg.crop,
            OUTPUT_STRIDE,
            norm_coords=cfg.norm_coords,
            wot=cfg.wot,
            wtv=cfg.wtv,
            wcount=cfg.wcount,
            waux=cfg.waux,
            aux_sigma=cfg.aux_sigma,
            dense_weight_alpha=cfg.dense_weight_alpha,
            ot_iterations=cfg.ot_iterations,
            reg=cfg.reg,
        ).to(self.device)
        self.miner = None
        if cfg.hard_mining_gamma > 0:
            self.miner = HardExampleTracker(
                [smp.name for smp in train],
                gamma=cfg.hard_mining_gamma,
                ema_decay=cfg.hard_mining_ema,
                clip=cfg.hard_mining_clip,
                min_count=cfg.hard_mining_min_count,
            )
        if resuming:
            self._resume()
        self.ema = ModelEMA(self.model, cfg.ema_decay) if cfg.ema else None
        if resuming and self.ema is not None:
            self.ema.load_state_dict(self._resume_state["ema_state_dict"])
        self.best_files = BestCheckpoints(self.run_dir, cfg.keep_best)

    def _scheduler(self):
        cfg = self.config
        if not cfg.scheduler:
            return None
        warmup = max(cfg.warmup_epochs, 1)
        return optim.lr_scheduler.SequentialLR(
            self.optimizer,
            [
                optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(cfg.epochs - warmup, 1), eta_min=cfg.lr * 0.01
                ),
            ],
            milestones=[warmup],
        )

    def _resume(self) -> None:
        state = torch.load(self.run_dir / "last.tar", map_location=self.device, weights_only=True)
        self._resume_state = state
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        # RNG states are CPU byte tensors; map_location moved them to the device
        self.rng.set_state(state["photometric_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.miner is not None and state.get("hard_mining_state"):
            self.miner.load_state_dict(state["hard_mining_state"])
        self.start_epoch = state["epoch"] + 1
        self._truncate_metrics(state["epoch"])
        self.best_score, self.evals_since_best = state["best_score"], state["evals_since_best"]
        self.best = state.get("best")
        self._refresh_sampling(self.start_epoch)
        logger.info("Resumed after epoch %d.", state["epoch"])

    def _truncate_metrics(self, last_epoch: int) -> None:
        """Drop metrics of epochs after the saved state: they are about to be run again."""
        path = self.run_dir / "metrics.jsonl"
        if path.is_file():
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept = [line for line in lines if json.loads(line)["epoch"] <= last_epoch]
            path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.config
        status, done, t0 = "finished", self.start_epoch - 1, time.perf_counter()
        try:
            for epoch in range(self.start_epoch, cfg.epochs):
                record = self.train_epoch(epoch)
                validated = epoch >= cfg.val_start and epoch % cfg.val_every == 0
                if validated:
                    record["val"] = self.validate(epoch)
                with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                if self.scheduler is not None:
                    self.scheduler.step()
                done = epoch
                if validated:
                    self._save_last(done)
                    if cfg.patience and self.evals_since_best >= cfg.patience:
                        status = "early_stop"
                        logger.info("Early stop: %d validations without a new best.", self.evals_since_best)
                        break
        except KeyboardInterrupt:
            status = "interrupted"
            logger.info("Interrupted; saving the state after epoch %d.", done)
        if done >= self.start_epoch:
            self._save_last(done)
        summary = {
            "status": status,
            "epochs_completed": done + 1,
            "seconds": round(time.perf_counter() - t0, 1),
            "best": self.best,
            "run_dir": str(self.run_dir),
            "code_version": _code_version(),
            "config": cfg.to_json(),
        }
        write_json(self.run_dir / "summary.json", summary)
        if self.best:
            logger.info(
                "Best: epoch %d, MAE %.2f, RMSE %.2f, counting accuracy %.1f%% -> %s",
                self.best["epoch"],
                self.best["mae"],
                self.best["rmse"],
                self.best["counting_accuracy"] * 100,
                self.best["checkpoint"],
            )
        return summary

    def train_epoch(self, epoch: int) -> dict:
        cfg, device = self.config, self.device
        self.model.train()
        self.sampler.set_epoch(epoch)
        sums = torch.zeros(len(PARTS) + 3, device=device)  # parts, loss, |err|, err^2
        crops, mined = 0, []
        accum, n_batches = cfg.accum_steps, len(self.loader)
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.loader):
            images = self.photometric(batch["image"].to(device, non_blocking=True), augment=True, generator=self.rng)
            target = batch["density"].to(device, non_blocking=True)
            points = [p.to(device, non_blocking=True) for p in batch["points"]]
            pred = self.model(images)
            total, parts = self.loss(pred, target, points)
            (total / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == n_batches:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

            with torch.no_grad():
                n = pred.shape[0]
                gt = target.flatten(1).sum(1)
                err = pred.detach().flatten(1).sum(1) - gt
                sums += torch.stack(
                    [*(parts[k] * n for k in PARTS), total.detach() * n, err.abs().sum(), err.pow(2).sum()]
                )
                if self.miner is not None:
                    mined.append((batch["name"], torch.stack([err, gt])))
            crops += pred.shape[0]

        values = (sums / crops).tolist()  # the epoch's one sync
        train = dict(zip(PARTS, values[: len(PARTS)], strict=True))
        train["loss"], train["mae"], train["rmse"] = values[len(PARTS)], values[-2], math.sqrt(values[-1])
        if self.miner is not None:
            for names, stats in mined:
                err, gt = stats.cpu().numpy()
                self.miner.observe(names, err, gt)
            self._refresh_sampling(epoch + 1)
        lr = self.optimizer.param_groups[0]["lr"]
        seconds = time.perf_counter() - t0
        logger.info(
            "Epoch %d train | lr %.2e | loss %.2f | count %.2f | tv %.3f | aux %.2f | wass %.3f | MAE %.2f | RMSE %.2f | %.1fs",
            epoch,
            lr,
            train["loss"],
            train["count"],
            train["tv"],
            train["aux"],
            train["wd"],
            train["mae"],
            train["rmse"],
            seconds,
        )
        return {"epoch": epoch, "lr": lr, "seconds": round(seconds, 3), "train": train}

    def _refresh_sampling(self, epoch: int) -> None:
        """From `hard_mining_start` on, sample by the tracked errors."""
        if self.miner is None or epoch < self.config.hard_mining_start:
            return
        weights = self.miner.weights()
        self.sampler.set_weights(weights)
        if epoch % self.config.val_every == 0:
            hardest = ", ".join(f"{n} {e:.2f}" for n, e in self.miner.hardest(5))
            logger.info(
                "Hard mining | weights %.2f-%.2f | seen %.0f%% | hardest: %s",
                weights.min(),
                weights.max(),
                100 * self.miner.coverage(),
                hardest,
            )

    # ------------------------------------------------------------------
    # Validation and checkpoints
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def validate(self, epoch: int) -> dict:
        t0 = time.perf_counter()
        weights = (self.ema.model if self.ema is not None else self.model).state_dict()
        net = DensityNet(freeze_backbone_bn=self.config.freeze_backbone_bn)
        net.load_state_dict(weights)
        net = prepare_inference_model(net, self.device, input_order="bgr")
        predictor = DensityPredictor(
            net, frame_size=self.settings.frame_size, masks=self.val_masks, max_batch=VAL_BATCH
        )
        preds, gts = [], []
        for result in evaluate(self.val_samples, predictor, self.cache, self.val_mask_ids, num_workers=0):
            preds.append(result.pred)
            gts.append(result.gt)
        preds, gts = np.array(preds), np.array(gts)
        m = compute_metrics(preds, gts)
        val = {
            "mae": m.mae,
            "rmse": m.rmse,
            "counting_accuracy": m.counting_accuracy,
            "primary_accuracy": fraction_within(preds, gts, abs_tol=PRIMARY_ABS_TOL, rel_tol=PRIMARY_REL_TOL),
            "bias": m.bias,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        s = runs.score(m.mae, m.rmse)
        improved = s < self.best_score
        if improved:
            self.best_score, self.evals_since_best = s, 0
            path = self.best_files.add(weights, epoch, m.mae, m.rmse)
            self.best = {"epoch": epoch, "checkpoint": str(path), **val}
        else:
            self.evals_since_best += 1
        val["best"] = improved
        logger.info(
            "Epoch %d val%s | MAE %.2f | RMSE %.2f | counting accuracy %.1f%% | Acc@(%g,%g%%) %.1f%% | %.1fs%s",
            epoch,
            " (EMA)" if self.ema is not None else "",
            m.mae,
            m.rmse,
            m.counting_accuracy * 100,
            PRIMARY_ABS_TOL,
            PRIMARY_REL_TOL * 100,
            val["primary_accuracy"] * 100,
            val["seconds"],
            "  * new best" if improved else f"  ({self.evals_since_best} without improvement)",
        )
        return val

    def _save_last(self, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "hard_mining_state": self.miner.state_dict() if self.miner is not None else None,
            "photometric_rng": self.rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "best_score": self.best_score,
            "evals_since_best": self.evals_since_best,
            "best": self.best,
        }
        save_atomic(state, self.run_dir / "last.tar")


def _code_version() -> str | None:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
