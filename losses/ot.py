"""Optimal-transport loss for crowd counting (DM-Count, Wang et al., NeurIPS 2020).

Computes per-image OT between the predicted (normalized) density map and the
empirical distribution over annotated keypoints. Sinkhorn solves for the dual
potential `beta`, then DM-Count's analytical gradient
    d(OT)/d(unnormed_density) = beta / sum(unnormed) - <beta, unnormed> / sum(unnormed)^2
is applied via a surrogate `loss = <unnormed_density, im_grad.detach()>`.
"""

import logging

import torch
from torch.nn import Module

from .sinkhorn import sinkhorn


logger = logging.getLogger(__name__)


class OTLoss(Module):
    def __init__(self, c_size, stride, norm_cood, num_of_iter_in_ot=100, reg=10.0):
        super().__init__()
        if c_size % stride != 0:
            raise ValueError(f"c_size ({c_size}) must be divisible by stride ({stride})")

        self.c_size = c_size
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # Pixel centers of each density-map cell, in input-image coordinates.
        # Registered as a buffer so the module moves with `.to(device)`.
        cood = torch.arange(0, c_size, step=stride, dtype=torch.float32) + stride / 2
        if norm_cood:
            cood = cood / c_size * 2 - 1  # map to [-1, 1]
        self.register_buffer("cood", cood.unsqueeze(0))
        self.output_size = self.cood.size(1)

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        if len(points) != batch_size:
            raise ValueError(f"points list length {len(points)} != batch size {batch_size}")
        if self.output_size != normed_density.size(2):
            raise ValueError(
                f"density map H ({normed_density.size(2)}) doesn't match expected output_size ({self.output_size})"
            )

        device = normed_density.device
        loss = torch.zeros((), device=device)
        ot_obj_values = torch.zeros((), device=device)
        wd = torch.zeros((), device=device)

        for idx, im_points in enumerate(points):
            if len(im_points) == 0:
                continue

            if self.norm_cood:
                im_points = im_points / self.c_size * 2 - 1
            x = im_points[:, 0:1]
            y = im_points[:, 1:2]
            x_dis = (x - self.cood).pow(2)  # (#gt, #cood)
            y_dis = (y - self.cood).pow(2)  # (#gt, #cood)
            dis = y_dis.unsqueeze(2) + x_dis.unsqueeze(1)
            dis = dis.view(dis.size(0), -1)  # (#gt, #cood^2)

            source_prob = normed_density[idx][0].view(-1).detach()
            target_prob = torch.full((len(im_points),), 1.0 / len(im_points), device=device)

            P, log = sinkhorn(
                target_prob,
                source_prob,
                dis,
                self.reg,
                max_iter=self.num_of_iter_in_ot,
                log=True,
            )
            if log.get("instability"):
                logger.warning("OTLoss: skipping image %d due to sinkhorn instability", idx)
                continue

            beta = log["beta"]  # (#cood^2,)
            ot_obj_values = (
                ot_obj_values + (normed_density[idx] * beta.view(1, self.output_size, self.output_size)).sum()
            )

            # Analytical gradient (DM-Count eq.) wrapped in a surrogate.
            source_density = unnormed_density[idx][0].view(-1).detach()
            source_count = source_density.sum().clamp_min(1e-8)
            im_grad = beta / source_count - (source_density * beta).sum() / (source_count * source_count)
            im_grad = im_grad.detach().view(1, self.output_size, self.output_size)
            loss = loss + (unnormed_density[idx] * im_grad).sum()

            # Accumulate Wasserstein distance on-device; caller .item()s it once.
            wd = wd + (dis * P).sum()

        return loss, wd, ot_obj_values
