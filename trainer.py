"""Training orchestration for the chicken-density model.

`Trainer.setup()` builds the data pipeline, model + EMA + scheduler, and the
combined DM-Count loss. `Trainer.train()` runs the train/val loop, saves a
checkpoint per epoch, and tracks a single best-model file by (2*MSE + MAE).
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import DOWNSAMPLE_RATIO, BirdDataset, collate, seed_worker
from hard_mining import HardExampleTracker
from losses import DMCountLoss
from models.ema import ModelEMA
from models.shufflenet import get_shufflenet_density_model
from utils import AverageMeter, Logger, SaveHandle


_LOSS_KEYS = ("loss", "ot", "count", "tv", "aux", "wd", "ot_obj", "mae", "mse")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.start_epoch = 0
        self.best_mae = float("inf")
        self.best_mse = float("inf")
        # Populated by `_setup_hard_mining`; both stay None when mining is off.
        self.hard_miner = None
        self.train_sampler = None
        # Consecutive validation runs since the last best-model improvement.
        # Reset on every new best; drives early stopping when patience is set.
        self.evals_since_improvement = 0

    # ----- setup -------------------------------------------------------------

    def setup(self):
        self._require_cuda()
        self._setup_save_dir()
        self._setup_logging()
        self._setup_data()
        self._setup_model_and_optim()
        resumed_ema_state = self._maybe_resume()
        self._setup_ema(resumed_ema_state)

        # Rolling caps on disk: 1 epoch checkpoint (.tar) + N best-model files (.pth).
        self.save_list = SaveHandle(max_num=1)
        self.best_save_list = SaveHandle(max_num=max(1, self.args.max_best_ckpts))

    def _require_cuda(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this trainer requires a GPU.")
        n = torch.cuda.device_count()
        if n != 1:
            raise RuntimeError(f"Expected exactly 1 visible GPU; found {n}. Set CUDA_VISIBLE_DEVICES to one device.")
        self.device = torch.device("cuda")

    def _setup_save_dir(self):
        a = self.args
        sub = (
            f"input-{a.crop_size}_wot-{a.wot}_wtv-{a.wtv}_waux-{a.waux}"
            f"_reg-{a.reg}_nIter-{a.num_of_iter_in_ot}_normCood-{int(a.norm_cood)}"
            f"/{time.strftime('%Y%m%d-%H%M%S')}"
        )
        self.save_dir = Path(a.checkpoint_dir) / sub
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        time_str = datetime.now().strftime("%m%d-%H%M%S")
        self.logger = Logger(str(self.save_dir / f"train-{time_str}.log"))
        self.logger.print_config(vars(self.args))
        self.logger.info("using GPU: %s", torch.cuda.get_device_name(0))

    def _setup_data(self):
        a = self.args
        self.datasets = {
            "train": BirdDataset(a.data_dir, a.crop_size, DOWNSAMPLE_RATIO, split="train", train_size=a.train_size),
            "val": BirdDataset(a.data_dir, a.crop_size, DOWNSAMPLE_RATIO, split="val", test_size=a.test_size),
        }
        self._setup_hard_mining()
        self.dataloaders = {
            "train": self._make_loader(
                "train", batch_size=a.batch_size, shuffle=True, pin_memory=True, sampler=self.train_sampler
            ),
            "val": self._make_loader("val", batch_size=1, shuffle=False, pin_memory=False),
        }

    def _setup_hard_mining(self):
        """Build the per-image error tracker + its sampler, if mining is on.

        Both stay None when `--hard-mining-gamma 0` (the default), leaving the
        train loader on plain `shuffle=True`.
        """
        a = self.args
        self.hard_miner = None
        self.train_sampler = None
        if getattr(a, "hard_mining_gamma", 0.0) <= 0.0:
            return

        names = [Path(p).stem for p in self.datasets["train"].im_list]
        self.hard_miner = HardExampleTracker(
            names,
            gamma=a.hard_mining_gamma,
            ema_decay=a.hard_mining_ema,
            clip=a.hard_mining_clip,
            min_count=a.hard_mining_min_count,
        )
        # Starts uniform; `_refresh_sampling_weights` swaps in error-derived
        # weights once epoch >= --hard-mining-start-epoch. `replacement=True` is
        # what lets a hard image appear several times in one epoch.
        self.train_sampler = WeightedRandomSampler(
            torch.ones(len(names), dtype=torch.double), num_samples=len(names), replacement=True
        )
        self.logger.info(
            "hard-example mining: gamma=%.2f, active from epoch %d, ema=%.3f, clip=%.1fx, %d train images",
            a.hard_mining_gamma,
            a.hard_mining_start_epoch,
            a.hard_mining_ema,
            a.hard_mining_clip,
            len(names),
        )

    def _make_loader(
        self, split: str, *, batch_size: int, shuffle: bool, pin_memory: bool, sampler=None
    ) -> DataLoader:
        nw = self.args.num_workers
        return DataLoader(
            self.datasets[split],
            batch_size=batch_size,
            # DataLoader rejects shuffle together with an explicit sampler; the
            # sampler already draws in random order.
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=pin_memory,
            persistent_workers=nw > 0,
            worker_init_fn=seed_worker,
        )

    def _setup_model_and_optim(self):
        a = self.args
        self.model = get_shufflenet_density_model(
            device=self.device,
            freeze_backbone_bn=not getattr(a, "no_freeze_backbone_bn", False),
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        self.scheduler = self._build_scheduler()
        self.loss_fn = DMCountLoss(
            a.crop_size,
            DOWNSAMPLE_RATIO,
            a.norm_cood,
            self.device,
            wot=a.wot,
            wtv=a.wtv,
            wcount=a.wcount,
            waux=a.waux,
            aux_sigma=a.aux_sigma,
            dense_weight_alpha=a.dense_weight_alpha,
            num_of_iter_in_ot=a.num_of_iter_in_ot,
            reg=a.reg,
        )

    def _build_scheduler(self):
        a = self.args
        if a.no_scheduler:
            return None
        warmup = max(a.warmup_epochs, 1)
        cosine = max(a.max_epoch - warmup, 1)
        return optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cosine, eta_min=a.lr * 0.01),
            ],
            milestones=[warmup],
        )

    def _maybe_resume(self) -> Optional[dict]:
        """Return the EMA state-dict captured from a `.tar` resume, or None."""
        path = self.args.resume
        if not path:
            self.logger.info("training from random init")
            return None

        self.logger.info("resuming from %s", path)
        suffix = Path(path).suffix.lower()
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if suffix == ".tar":
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            sched_state = ckpt.get("scheduler_state_dict")
            if self.scheduler is not None and sched_state is not None:
                self.scheduler.load_state_dict(sched_state)
            self.start_epoch = ckpt["epoch"] + 1
            self.logger.info("resumed at epoch %d (next training epoch)", self.start_epoch)
            self._restore_hard_mining(ckpt.get("hard_mining_state"))
            return ckpt.get("ema_state_dict")
        if suffix == ".pth":
            self.model.load_state_dict(ckpt)
            self.logger.info("resumed model weights only (.pth); optimizer/scheduler/EMA re-initialized")
            return None
        raise ValueError(f"unknown checkpoint extension {suffix!r}; expected .tar or .pth")

    def _restore_hard_mining(self, state: Optional[dict]):
        """Carry the per-image error EMA across a resume, so mining does not
        restart from a cold (uniform) tracker mid-run."""
        if self.hard_miner is None:
            if state is not None:
                self.logger.info("hard mining disabled; ignoring tracker state from checkpoint")
            return
        if state is None:
            self.logger.info("checkpoint carries no hard-mining state; tracker starts cold")
            return
        matched = self.hard_miner.load_state_dict(state)
        self.logger.info("restored hard-mining errors for %d/%d train images", matched, len(self.hard_miner))
        self._refresh_sampling_weights(self.start_epoch, log=True)

    def _setup_ema(self, resumed_ema_state: Optional[dict]):
        if not self.args.ema:
            self.ema = None
            if resumed_ema_state is not None:
                self.logger.info("--ema not set; ignoring EMA state from checkpoint")
            return
        # EMA must be created AFTER weights are loaded — it deepcopies the model.
        self.ema = ModelEMA(self.model, decay=self.args.ema_decay)
        if resumed_ema_state is not None:
            self.ema.ema.load_state_dict(resumed_ema_state)

    # ----- training loop -----------------------------------------------------

    def train(self):
        for epoch in range(self.start_epoch, self.args.max_epoch + 1):
            self.epoch = epoch
            self.logger.info("---- Epoch %d/%d ----", epoch, self.args.max_epoch)
            self.train_epoch()
            if self._should_validate():
                self.val_epoch()
                if self._should_early_stop():
                    self.logger.info(
                        "Early stopping at epoch %d: %d consecutive eval(s) without improvement (patience=%d).",
                        epoch,
                        self.evals_since_improvement,
                        self.args.patience,
                    )
                    break
            if self.scheduler is not None:
                self.scheduler.step()

    def _should_validate(self) -> bool:
        a = self.args
        return self.epoch >= a.val_start and (self.epoch % a.val_epoch == 0)

    def _should_early_stop(self) -> bool:
        patience = self.args.patience or 0
        return patience > 0 and self.evals_since_improvement >= patience

    def train_epoch(self):
        self.model.train()
        meters = {k: AverageMeter() for k in _LOSS_KEYS}
        t0 = time.time()
        accum_steps = max(1, self.args.accum_steps)
        loader = self.dataloaders["train"]
        n_batches = len(loader)

        self.optimizer.zero_grad(set_to_none=True)
        for step, sample in enumerate(loader):
            inputs = sample["image"].to(self.device, non_blocking=True)
            points = [p.to(self.device, non_blocking=True) for p in sample["keypoints"]]
            gt_density = sample["density"].to(self.device, non_blocking=True)
            N = inputs.size(0)

            names = sample["name"]

            outputs = self.model(inputs)
            total, parts = self.loss_fn(outputs, gt_density, points)

            # Scale so the accumulated gradient is the *mean* across micro-batches.
            (total / accum_steps).backward()

            # Step the optimizer every `accum_steps` batches, plus a final flush
            # at end-of-epoch for any partial accumulation.
            if ((step + 1) % accum_steps == 0) or ((step + 1) == n_batches):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

            self._update_meters(meters, total, parts, outputs, points, N, names)

        self.logger.info(self._format_train_log(meters, time.time() - t0))
        # Weights computed now take effect on the next pass over the loader.
        self._refresh_sampling_weights(self.epoch + 1)
        self._save_epoch_checkpoint()

    def _update_meters(self, meters, total, parts, outputs, points, N, names=None):
        with torch.no_grad():
            pred_count = outputs.view(N, -1).sum(dim=1)
            gd_count = torch.tensor([len(p) for p in points], device=self.device, dtype=torch.float32)
            err = (pred_count - gd_count).cpu().numpy()
            gt = gd_count.cpu().numpy()
        meters["loss"].update(total.item(), N)
        for k, v in parts.items():
            meters[k].update(v, N)
        meters["mae"].update(float(np.mean(np.abs(err))), N)
        meters["mse"].update(float(np.mean(err * err)), N)
        if self.hard_miner is not None and names is not None:
            # Errors are per *crop*, hence noisy; the tracker smooths them.
            self.hard_miner.observe(names, err, gt)

    def _refresh_sampling_weights(self, epoch: int, *, log: bool = False):
        """Push the tracker's current weights into the train sampler.

        Before `--hard-mining-start-epoch` this is a no-op on purpose: the
        tracker keeps accumulating error while the model is still too raw for
        "hardest" to mean anything but "most birds".
        """
        if self.hard_miner is None or epoch < self.args.hard_mining_start_epoch:
            return

        w = self.hard_miner.weights()
        self.train_sampler.weights = torch.as_tensor(w, dtype=torch.double)
        if log or epoch % self.args.val_epoch == 0:
            hardest = ", ".join(f"{n} {e:.2f}" for n, e in self.hard_miner.hardest(5))
            self.logger.info(
                "Hard mining (epoch %d) | weight %.2f-%.2f | seen %.0f%% | hardest: %s",
                epoch,
                float(w.min()),
                float(w.max()),
                100.0 * self.hard_miner.coverage(),
                hardest,
            )

    def _format_train_log(self, meters, dt: float) -> str:
        lr = self.optimizer.param_groups[0]["lr"]
        return (
            f"Epoch {self.epoch} Train | lr {lr:.2e} | loss {meters['loss'].avg:.2f} "
            f"| ot {meters['ot'].avg:.2e} | wass {meters['wd'].avg:.2f} "
            f"| count {meters['count'].avg:.2f} | tv {meters['tv'].avg:.2f} "
            f"| aux {meters['aux'].avg:.2f} "
            f"| MSE {np.sqrt(meters['mse'].avg):.2f} | MAE {meters['mae'].avg:.2f} "
            f"| {dt:.1f}s"
        )

    @torch.inference_mode()
    def val_epoch(self):
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        residuals = []
        t0 = time.time()

        for sample in self.dataloaders["val"]:
            inputs = sample["image"].to(self.device, non_blocking=True)
            if inputs.size(0) != 1:
                raise RuntimeError(f"val batch size must be 1, got {inputs.size(0)}")
            outputs = eval_model(inputs)
            residuals.append(sample["density"].sum().item() - outputs.sum().item())

        residuals = np.asarray(residuals)
        mse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        tag = "EMA" if self.ema is not None else "online"
        self.logger.info(
            "Epoch %d Val (%s) | MSE %.2f | MAE %.2f | %.1fs",
            self.epoch,
            tag,
            mse,
            mae,
            time.time() - t0,
        )

        if self._is_best(mse, mae):
            self.best_mse, self.best_mae = mse, mae
            self.evals_since_improvement = 0
            best_path = self.save_dir / f"best_ep{self.epoch:04d}_mae{mae:.2f}_mse{mse:.2f}.pth"
            torch.save(eval_model.state_dict(), best_path)
            self.best_save_list.append(best_path)  # rotates older best_*.pth out
            self.logger.info("saved best -> %s", best_path.name)
        else:
            self.evals_since_improvement += 1
            self.logger.info(
                "no improvement for %d consecutive eval(s) (patience=%d)",
                self.evals_since_improvement,
                self.args.patience or 0,
            )

    def _is_best(self, mse: float, mae: float) -> bool:
        return (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae)

    # ----- checkpoints -------------------------------------------------------

    def _save_epoch_checkpoint(self):
        path = self.save_dir / f"{self.epoch}_ckpt.tar"
        ckpt = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler is not None else None),
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.ema.state_dict()
        if self.hard_miner is not None:
            ckpt["hard_mining_state"] = self.hard_miner.state_dict()
        torch.save(ckpt, path)
        self.save_list.append(str(path))
