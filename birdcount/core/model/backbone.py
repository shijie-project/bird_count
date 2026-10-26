"""ShuffleNetV2-1.0x, stem through stage3 — the density model's backbone.

Ported from torchvision (`torchvision/models/shufflenetv2.py`, BSD-3-Clause,
Copyright (c) Soumith Chintala 2016, all rights reserved) so that loading the
model never imports torchvision: that import alone costs every model-loading
process ~1 s of startup and ~100 MB RSS, and ties deployments to a
torchvision build matching their CUDA torch.

Only the layers the density model uses are kept (conv1, maxpool, stage2,
stage3; no stage4 / conv5 / classifier). Module and parameter names are
identical to torchvision's, so checkpoints and torchvision's ImageNet weights
load unchanged; `tests/core/model/test_backbone.py` checks both against
torchvision itself.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# torchvision's ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1. The filename carries
# the hash prefix torch.hub checks, and the download lands in the same cache
# file torchvision uses.
IMAGENET_WEIGHTS_URL = "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth"

STEM_CHANNELS = 24
STAGE_REPEATS = (4, 8)  # stage2, stage3
STAGE_CHANNELS = (116, 232)  # stage2, stage3 output channels


def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


def _depthwise_conv(channels: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(channels, channels, 3, stride, 1, bias=False, groups=channels)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__()
        self.stride = stride
        branch = oup // 2
        if stride == 1 and inp != branch * 2:
            raise ValueError(f"stride 1 needs inp == oup (got {inp} -> {oup})")

        if stride > 1:
            self.branch1 = nn.Sequential(
                _depthwise_conv(inp, stride),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch, 1, bias=False),
                nn.BatchNorm2d(branch),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride > 1 else branch, branch, 1, bias=False),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
            _depthwise_conv(branch, stride),
            nn.BatchNorm2d(branch),
            nn.Conv2d(branch, branch, 1, bias=False),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return _channel_shuffle(out, 2)


def _stage(inp: int, oup: int, repeats: int) -> nn.Sequential:
    return nn.Sequential(InvertedResidual(inp, oup, 2), *(InvertedResidual(oup, oup, 1) for _ in range(repeats - 1)))


class ShuffleNetV2Backbone(nn.Module):
    """forward(x) -> (stage2 features at 1/8, stage3 features at 1/16)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, STEM_CHANNELS, 3, 2, 1, bias=False),
            nn.BatchNorm2d(STEM_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        c2, c3 = STAGE_CHANNELS
        self.stage2 = _stage(STEM_CHANNELS, c2, STAGE_REPEATS[0])
        self.stage3 = _stage(c2, c3, STAGE_REPEATS[1])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        f2 = self.stage2(self.maxpool(self.conv1(x)))
        return f2, self.stage3(f2)

    def load_imagenet_weights(self) -> None:
        """Initialise from torchvision's ImageNet weights (training from scratch only)."""
        state = torch.hub.load_state_dict_from_url(IMAGENET_WEIGHTS_URL, map_location="cpu", check_hash=True)
        own = self.state_dict().keys()
        self.load_state_dict({k: v for k, v in state.items() if k in own}, strict=True)
