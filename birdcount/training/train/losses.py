"""The training loss: DM-Count (OT + count + normalised L1) plus a dense Gaussian term.

    total = wot * OT + wcount * |sum(pred) - N| + wtv * TV + waux * aux

* OT: distribution match against the annotated points (`ot.OTLoss`), mean
  over the batch.
* count: L1 between predicted and annotated count.
* TV: L1 between the normalised predicted and point-count maps, weighted by
  the image's count.
* aux: per-pixel L1 against the point-count map blurred by a Gaussian of
  `aux_sigma` cells. The DM-Count terms say nothing about empty areas (no
  points: no OT, no TV); this one supervises every pixel, which steadies
  early training and suppresses spurious density in sparse frames.
  `dense_weight_alpha` > 0 weights pixels by (1 + alpha * target).

Same terms, weights and reductions as v2, so its hyperparameters carry over.
Returns the total and its weighted parts as tensors: nothing here syncs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from birdcount.training.train.ot import OTLoss


class GaussianSmooth(nn.Module):
    """Separable 2-D Gaussian blur of (B, 1, H, W) maps (two 1-D convs)."""

    def __init__(self, sigma: float):
        super().__init__()
        size = int(2 * math.ceil(3 * sigma) + 1)
        half = size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        k = torch.exp(-(x**2) / (2.0 * sigma * sigma))
        k = k / k.sum()
        self.register_buffer("kernel_h", k.view(1, 1, 1, size))
        self.register_buffer("kernel_v", k.view(1, 1, size, 1))
        self.pad = half

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(x, self.kernel_h, padding=(0, self.pad))
        return F.conv2d(x, self.kernel_v, padding=(self.pad, 0))


class DMCountLoss(nn.Module):
    def __init__(
        self,
        crop: int,
        stride: int,
        *,
        norm_coords: bool,
        wot: float = 0.1,
        wtv: float = 0.01,
        wcount: float = 1.0,
        waux: float = 1.0,
        aux_sigma: float = 2.0,
        dense_weight_alpha: float = 0.0,
        ot_iterations: int = 100,
        reg: float = 10.0,
    ):
        super().__init__()
        self.ot = OTLoss(crop, stride, norm_coords, ot_iterations, reg)
        self.smooth = GaussianSmooth(aux_sigma) if aux_sigma > 0 else None
        self.wot, self.wtv, self.wcount, self.waux = wot, wtv, wcount, waux
        self.alpha = dense_weight_alpha

    def forward(self, pred: Tensor, target: Tensor, points: list[Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """pred / target: (B, 1, S, S) predicted density / point counts per cell; points: (N_i, 2) per image."""
        n = pred.shape[0]
        pred_count = pred.flatten(1).sum(1)
        normed = pred / (pred_count.view(n, 1, 1, 1) + 1e-6)

        ot, wd, ot_obj = self.ot(normed, pred, points)
        ot, wd, ot_obj = ot / n, wd / n, ot_obj / n

        gt_count = torch.tensor([len(p) for p in points], device=pred.device, dtype=pred.dtype)
        count = (pred_count - gt_count).abs().mean()

        target_normed = target / (gt_count.view(n, 1, 1, 1) + 1e-6)
        tv = ((normed - target_normed).abs().flatten(1).sum(1) * gt_count).mean()

        aux = pred.new_zeros(())
        if self.smooth is not None:
            blurred = self.smooth(target)
            diff = (pred - blurred).abs()
            if self.alpha > 0:
                diff = diff * (1.0 + self.alpha * blurred)
            aux = diff.flatten(1).sum(1).mean()

        total = self.wot * ot + self.wcount * count + self.wtv * tv + self.waux * aux
        parts = {
            "ot": self.wot * ot.detach(),
            "count": self.wcount * count.detach(),
            "tv": self.wtv * tv.detach(),
            "aux": self.waux * aux.detach(),
            "wd": wd.detach(),
            "ot_obj": self.wot * ot_obj.detach(),
        }
        return total, parts
