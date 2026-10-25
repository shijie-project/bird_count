"""The density network: ShuffleNetV2 backbone + U-Net-style skip head.

    forward(x: [B, 3, H, W]) -> density: [B, 1, H/8, W/8], non-negative

The head fuses stage2 features (1/8) with upsampled stage3 features (1/16)
so it can localise small birds precisely while still using the deeper
features. H and W must be multiples of 8 (`OUTPUT_STRIDE`); not of 16, as
stage3 is resized to stage2's exact size rather than scaled by 2.

Attribute names (`backbone`, `reg_layer`, `density_conv`) are the checkpoint
format: renaming one breaks loading every existing checkpoint.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from birdcount.core.model.backbone import STAGE_CHANNELS, ShuffleNetV2Backbone


logger = logging.getLogger(__name__)

OUTPUT_STRIDE = 8

InputOrder = Literal["rgb", "bgr"]


class DensityNet(nn.Module):
    def __init__(self, *, freeze_backbone_bn: bool = True):
        super().__init__()
        self.backbone = ShuffleNetV2Backbone()
        self.freeze_backbone_bn = freeze_backbone_bn
        self.reg_layer = nn.Sequential(
            nn.Conv2d(sum(STAGE_CHANNELS), 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.density_conv = nn.Conv2d(128, 1, 1)
        # Channel order the first conv expects; see `set_input_order`.
        self.input_order: InputOrder = "rgb"

    def forward(self, x: Tensor) -> Tensor:
        f2, f3 = self.backbone(x)
        # Size-matched, not scale_factor=2: stage2 is odd-sized whenever the
        # input is a multiple of 8 but not 16 (1080 -> 135, stage3 68 -> 136).
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reg_layer(torch.cat([f2, f3], dim=1))
        return F.relu_(self.density_conv(x))

    def train(self, mode: bool = True) -> DensityNet:
        """Like nn.Module.train, but backbone BN keeps its ImageNet statistics
        when `freeze_backbone_bn` is set."""
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    # ------------------------------------------------------------------
    # Inference-time transforms (exact; applied once at load time)
    # ------------------------------------------------------------------

    def fuse_conv_bn(self) -> None:
        """Fold every BatchNorm into the conv before it. Eval mode only."""
        if self.training:
            raise RuntimeError("fuse_conv_bn needs eval mode: BN running stats must be frozen")
        before = sum(isinstance(m, nn.BatchNorm2d) for m in self.modules())
        _fuse_sequentials(self)
        left = sum(isinstance(m, nn.BatchNorm2d) for m in self.modules())
        if left:
            raise RuntimeError(f"{left} BatchNorm2d layers were not preceded by a Conv2d and stayed unfused")
        logger.debug("Fused %d Conv+BN pairs.", before)

    def set_input_order(self, order: InputOrder) -> None:
        """Make the model consume `order` channel order directly.

        Frames from cv2 / cameras are BGR; the model was trained on RGB.
        Swapping the first conv's input channels once is exactly equivalent
        to flipping every input, and saves a full-frame copy per batch.
        Normalisation mean/std must then be given in the same order.
        Idempotent: the current order is recorded on the model.
        """
        if order not in ("rgb", "bgr"):
            raise ValueError(f"input order must be 'rgb' or 'bgr', got {order!r}")
        if order == self.input_order:
            return
        conv = self.backbone.conv1[0]
        with torch.no_grad():
            conv.weight.copy_(conv.weight[:, [2, 1, 0]])
        self.input_order = order


def _fuse_sequentials(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            layers, kids, i = [], list(child), 0
            while i < len(kids):
                if i + 1 < len(kids) and isinstance(kids[i], nn.Conv2d) and isinstance(kids[i + 1], nn.BatchNorm2d):
                    layers.append(fuse_conv_bn_eval(kids[i], kids[i + 1]))
                    i += 2
                else:
                    _fuse_sequentials(kids[i])
                    layers.append(kids[i])
                    i += 1
            setattr(module, name, nn.Sequential(*layers))
        else:
            _fuse_sequentials(child)
