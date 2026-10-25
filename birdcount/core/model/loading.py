"""Building the density model and loading checkpoints.

Every consumer gets its model from here:

    load_inference_model(ckpt, device, input_order="bgr")   deploy, evaluation, video tools
    build_model(pretrained_backbone=True)                     training from scratch
    build_model(ckpt)                                         fine-tuning / resuming

so the inference model is prepared identically everywhere: eval mode,
Conv+BN fused, fp16 weights on CUDA, NCHW, gradients off.

Loading fails loudly. A missing checkpoint, or one whose keys do not match the
network, raises rather than falling back to random weights — a model that
silently counts noise is worse than one that does not start.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from birdcount.core.model.network import DensityNet, InputOrder


logger = logging.getLogger(__name__)

# Training checkpoints wrap the weights; EMA weights are preferred when present
# (they are what validation scored).
_WRAPPED_KEYS = ("ema_state_dict", "model_state_dict", "state_dict")


def read_state_dict(path: str | os.PathLike[str]) -> dict[str, Tensor]:
    """Model weights from a raw or wrapped checkpoint, on CPU.

    Uses `weights_only=True`: unpickling a checkpoint cannot execute code.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    state, source = checkpoint, "raw"
    if isinstance(checkpoint, Mapping):
        for key in _WRAPPED_KEYS:
            if checkpoint.get(key) is not None:
                state, source = checkpoint[key], key
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"{path.name} holds no state dict (got {type(state).__name__})")

    # DataParallel / DDP prefix every key with "module.".
    state = {k.removeprefix("module."): v for k, v in state.items()}
    logger.debug("Read %d tensors from %s (%s).", len(state), path.name, source)
    return state


def build_model(
    checkpoint: str | os.PathLike[str] | None = None,
    *,
    pretrained_backbone: bool = False,
    freeze_backbone_bn: bool = True,
) -> DensityNet:
    """A trainable DensityNet on CPU, in train mode.

    `checkpoint` loads every weight (strictly); `pretrained_backbone` instead
    starts the backbone from ImageNet and the head from scratch.
    """
    if checkpoint is not None and pretrained_backbone:
        raise ValueError("pass a checkpoint or pretrained_backbone=True, not both")
    model = DensityNet(freeze_backbone_bn=freeze_backbone_bn)
    if checkpoint is not None:
        model.load_state_dict(read_state_dict(checkpoint), strict=True)
    elif pretrained_backbone:
        model.backbone.load_imagenet_weights()
    return model.train()


def load_inference_model(
    checkpoint: str | os.PathLike[str],
    device: str | torch.device,
    *,
    input_order: InputOrder = "rgb",
    half: bool | None = None,
) -> DensityNet:
    """The inference-ready model: eval, fused, frozen, on `device`.

    `half` defaults to True on CUDA: fp16 weights with fp16 inputs (no
    autocast) measured 14 % faster than v2's autocast + channels_last path at
    batch 20 / 720p, with the same count error against fp32 (~0.6 birds max
    on 200). NCHW is kept: channels_last measured slower for this network
    (ShuffleNet's channel shuffle) on the RTX 5090 it was profiled on.
    """
    device = torch.device(device)
    if half is None:
        half = device.type == "cuda"
    model = build_model(checkpoint).eval()
    model.fuse_conv_bn()
    model.set_input_order(input_order)
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.float16 if half else torch.float32)
    logger.info(
        "Loaded %s on %s (%s, %s input).", Path(checkpoint).name, device, "fp16" if half else "fp32", input_order
    )
    return model
