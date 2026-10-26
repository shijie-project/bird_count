"""Every training hyperparameter, defined once.

`TrainConfig` is the single definition: the command line is generated from
its fields (so the web UI, which builds its form from `build_parser()`, sees
the same options), every run writes it to `config.json`, and `--resume`
reads it back, so a resumed run cannot silently differ from the original.

Defaults are the recipe of the production model (best_ep0660, Oct 2026):
DM-Count weights, EMA 0.996, normalised OT coordinates, 1000 epochs at lr
1e-5 with 5 warm-up epochs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path


def _meta(help: str, group: str) -> dict:
    return {"help": help, "group": group}


@dataclass(frozen=True)
class TrainConfig:
    # --- data ---------------------------------------------------------------------
    data_root: Path | None = field(default=None, metadata=_meta("dataset root (default: DATA_ROOT)", "data"))
    output_dir: Path | None = field(
        default=None, metadata=_meta("run folder (default: OUTPUT_ROOT/train/<time>[-name])", "data")
    )
    name: str = field(default="", metadata=_meta("suffix of the default run folder", "data"))
    crop: int = field(default=512, metadata=_meta("training crop size (multiple of 8)", "data"))
    train_sizes: tuple[int, ...] = field(
        default=(1280,), metadata=_meta("longest edge a frame is resized to before cropping", "data")
    )
    batch_size: int = field(default=16, metadata=_meta("crops per batch", "data"))
    accum_steps: int = field(
        default=1, metadata=_meta("batches per optimizer step (effective batch = batch_size x accum_steps)", "data")
    )
    num_workers: int = field(default=2, metadata=_meta("DataLoader worker processes", "data"))

    # --- optimisation -----------------------------------------------------------------
    lr: float = field(default=1e-5, metadata=_meta("peak learning rate", "optimization"))
    weight_decay: float = field(default=1e-4, metadata=_meta("AdamW weight decay", "optimization"))
    epochs: int = field(default=1000, metadata=_meta("training epochs", "optimization"))
    warmup_epochs: int = field(
        default=5, metadata=_meta("linear warm-up before cosine decay to lr/100", "optimization")
    )
    scheduler: bool = field(
        default=True, metadata=_meta("warm-up + cosine schedule (off: constant lr)", "optimization")
    )
    ema: bool = field(
        default=True, metadata=_meta("keep an EMA of the weights; it is what gets validated and saved", "optimization")
    )
    ema_decay: float = field(default=0.996, metadata=_meta("EMA decay per optimizer step", "optimization"))
    freeze_backbone_bn: bool = field(
        default=True, metadata=_meta("keep the backbone's BatchNorm at its ImageNet statistics", "optimization")
    )
    keep_best: int = field(default=5, metadata=_meta("best checkpoints kept on disk", "optimization"))

    # --- loss -------------------------------------------------------------------------
    wot: float = field(default=0.1, metadata=_meta("OT loss weight", "loss"))
    wtv: float = field(default=0.01, metadata=_meta("normalised-L1 (TV) weight", "loss"))
    wcount: float = field(default=1.0, metadata=_meta("count L1 weight", "loss"))
    waux: float = field(default=1.0, metadata=_meta("Gaussian-density aux weight", "loss"))
    aux_sigma: float = field(
        default=2.0, metadata=_meta("aux Gaussian sigma in density cells (0: no aux loss)", "loss")
    )
    dense_weight_alpha: float = field(default=0.0, metadata=_meta("weight aux pixels by 1 + alpha * target", "loss"))
    reg: float = field(default=10.0, metadata=_meta("Sinkhorn entropic regularisation", "loss"))
    ot_iterations: int = field(default=100, metadata=_meta("Sinkhorn iterations", "loss"))
    norm_coords: bool = field(
        default=True, metadata=_meta("OT distances on coordinates normalised to [-1, 1]", "loss")
    )

    # --- hard-example mining ----------------------------------------------------------
    hard_mining_gamma: float = field(
        default=0.0, metadata=_meta("oversample images by (relative error)^gamma; 0: off", "hard mining")
    )
    hard_mining_start: int = field(
        default=50, metadata=_meta("epoch from which sampling follows the errors", "hard mining")
    )
    hard_mining_ema: float = field(default=0.9, metadata=_meta("EMA decay of each image's error", "hard mining"))
    hard_mining_clip: float = field(
        default=3.0, metadata=_meta("cap on a weight's ratio to uniform, both ways", "hard mining")
    )
    hard_mining_min_count: float = field(
        default=5.0, metadata=_meta("floor of the GT count dividing the error", "hard mining")
    )

    # --- validation -------------------------------------------------------------------
    val_every: int = field(default=5, metadata=_meta("validate every N epochs", "validation"))
    val_start: int = field(default=30, metadata=_meta("first epoch that is validated", "validation"))
    patience: int = field(
        default=0, metadata=_meta("stop after N validations without a new best; 0: never", "validation")
    )
    masks: bool = field(
        default=True, metadata=_meta("validate with the cameras' region masks, as deployed", "validation")
    )

    # --- run --------------------------------------------------------------------------
    seed: int = field(default=42, metadata=_meta("seed for sampling, augmentation and initialisation", "run"))
    deterministic: bool = field(default=False, metadata=_meta("deterministic cuDNN kernels (slower)", "run"))
    device: str = field(default="cuda", metadata=_meta("torch device", "run"))
    init: Path | None = field(
        default=None, metadata=_meta("start from these weights (fine-tuning) instead of ImageNet", "run")
    )
    resume: Path | None = field(
        default=None, metadata=_meta("continue the run in this folder, with its config", "run")
    )

    def __post_init__(self):
        if self.crop % 8:
            raise ValueError(f"crop must be a multiple of 8, got {self.crop}")
        for name in ("batch_size", "accum_steps", "epochs", "val_every", "keep_best"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0 <= self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")

    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_json(cls, data: dict) -> TrainConfig:
        known = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"unknown config key(s) {sorted(unknown)}")
        hints = typing.get_type_hints(cls)
        kwargs = {}
        for name, value in data.items():
            hint = hints[name]
            if value is not None and Path in typing.get_args(hint):
                value = Path(value)
            elif value is not None and typing.get_origin(hint) is tuple:
                value = tuple(value)
            kwargs[name] = value
        return cls(**kwargs)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def build_parser() -> argparse.ArgumentParser:
    """The command line, one flag per `TrainConfig` field (`--wot`, `--no-ema`, ...)."""
    parser = argparse.ArgumentParser(prog="python -m birdcount.training.train", description="Train the density model.")
    hints = typing.get_type_hints(TrainConfig)
    groups: dict[str, argparse._ArgumentGroup] = {}
    for f in dataclasses.fields(TrainConfig):
        group = groups.setdefault(f.metadata["group"], parser.add_argument_group(f.metadata["group"]))
        flag = "--" + f.name.replace("_", "-")
        hint = hints[f.name]
        kw = {"default": f.default, "help": f"{f.metadata['help']} (default: {f.default})"}
        if hint is bool:
            group.add_argument(flag, action=argparse.BooleanOptionalAction, **kw)
        elif typing.get_origin(hint) is tuple:
            group.add_argument(flag, type=typing.get_args(hint)[0], nargs="+", **kw)
        elif Path in typing.get_args(hint):
            group.add_argument(flag, type=Path, **kw)
        else:
            group.add_argument(flag, type=hint, **kw)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    values = vars(args)
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return TrainConfig(**values)
