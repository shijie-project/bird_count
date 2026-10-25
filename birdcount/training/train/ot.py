"""Optimal-transport loss (DM-Count, Wang et al., NeurIPS 2020), batched.

Per image, entropic OT between the predicted density (normalised to a
distribution over cells) and the uniform distribution over annotated points.
Sinkhorn gives the dual potential `beta` over cells; DM-Count's analytical
gradient of the OT cost w.r.t. the unnormalised density,

    d(OT)/d(density) = beta / sum(density) - <beta, density> / sum(density)^2,

is applied through the surrogate `<density, grad.detach()>`.

v2 solved one image at a time and checked every iteration for NaN with a
host round trip: ~1700 GPU->CPU syncs per batch of 16, 66 of the 87.6 ms of
a training step. Here the whole batch is one problem — points padded with
zero mass — and stability and convergence are tracked per image on the
device, so a batch syncs only at the convergence checks (every
`CHECK_EVERY` iterations, while any image is still iterating). The rules are
v2's: an image stops at the first check where its marginal error is below
`stop_thr`; an image that turns non-finite rolls back one iteration and is
left out of the loss.

Sinkhorn after POT's `ot.bregman.sinkhorn` (MIT; M. Cuturi, "Sinkhorn
Distances", NeurIPS 2013).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


M_EPS = 1e-16
CHECK_EVERY = 10


def sinkhorn(a: Tensor, b: Tensor, cost: Tensor, reg: float, max_iter: int, stop_thr: float = 1e-9):
    """Batched Sinkhorn-Knopp.

    a: (B, N) point marginals (0 on padding), b: (B, M) cell marginals,
    cost: (B, N, M). Returns (u, v, K, unstable) with K = exp(-cost / reg)
    and `unstable` (B,) bool.
    """
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")
    batch, _, m = cost.shape
    kernel = torch.exp(cost / -reg)
    kernel_t = kernel.transpose(1, 2)
    u = a.clone()  # 1 / N on real points, as v2 starts; 0 on padding
    v = torch.full((batch, m), 1.0 / m, dtype=b.dtype, device=b.device)
    active = torch.ones(batch, dtype=torch.bool, device=b.device)
    unstable = torch.zeros_like(active)
    for it in range(max_iter):
        v_new = b / (torch.bmm(kernel_t, u.unsqueeze(2)).squeeze(2) + M_EPS)
        u_new = a / (torch.bmm(kernel, v_new.unsqueeze(2)).squeeze(2) + M_EPS)
        finite = torch.isfinite(u_new).all(1) & torch.isfinite(v_new).all(1)
        unstable |= active & ~finite  # rolled back: keeps the previous u, v
        active &= finite
        u = torch.where(active[:, None], u_new, u)
        v = torch.where(active[:, None], v_new, v)
        if it % CHECK_EVERY == 0:
            b_hat = torch.bmm(kernel_t, u.unsqueeze(2)).squeeze(2) * v
            active &= (b - b_hat).pow(2).sum(1) > stop_thr
            if not bool(active.any()):  # the only host sync: once per check
                break
    return u, v, kernel, unstable


class OTLoss(nn.Module):
    def __init__(self, crop: int, stride: int, norm_coords: bool, iterations: int = 100, reg: float = 10.0):
        super().__init__()
        if crop % stride:
            raise ValueError(f"crop ({crop}) must be divisible by stride ({stride})")
        self.crop = crop
        self.norm_coords = norm_coords
        self.iterations = iterations
        self.reg = reg
        # Centres of the density cells, in crop pixels (or [-1, 1]).
        coords = torch.arange(0, crop, step=stride, dtype=torch.float32) + stride / 2
        if norm_coords:
            coords = coords / crop * 2 - 1
        self.register_buffer("coords", coords)
        self.cells = len(coords)

    def forward(self, normed: Tensor, density: Tensor, points: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Sums over the batch of (surrogate loss, Wasserstein distance, OT objective).

        normed / density: (B, 1, S, S) predicted density, normalised / raw;
        points: per image (N_i, 2) (x, y) in crop pixels. Images without
        points (and unstable ones) contribute nothing.
        """
        if len(points) != normed.shape[0]:
            raise ValueError(f"{len(points)} point sets for a batch of {normed.shape[0]}")
        if normed.shape[-1] != self.cells or normed.shape[-2] != self.cells:
            raise ValueError(f"density map {tuple(normed.shape[-2:])} != {self.cells}x{self.cells} for this crop")
        zero = normed.new_zeros(())
        idx = [i for i, p in enumerate(points) if len(p)]
        if not idx:
            return zero, zero, zero

        pts = pad_sequence([points[i].to(normed.dtype) for i in idx], batch_first=True)  # (B', N, 2)
        counts = torch.tensor([len(points[i]) for i in idx], device=normed.device)
        real = torch.arange(pts.shape[1], device=normed.device)[None] < counts[:, None]  # (B', N)
        if self.norm_coords:
            pts = pts / self.crop * 2 - 1
        dx = (pts[..., 0:1] - self.coords).pow(2)  # (B', N, S)
        dy = (pts[..., 1:2] - self.coords).pow(2)
        cost = (dy.unsqueeze(3) + dx.unsqueeze(2)).flatten(2) * real.unsqueeze(2)  # (B', N, S*S), row-major y
        a = real.to(normed.dtype) / counts[:, None].to(normed.dtype)

        sel = torch.tensor(idx, device=normed.device)
        b = normed.index_select(0, sel).flatten(1).detach()  # (B', S*S)
        u, v, kernel, unstable = sinkhorn(a, b, cost, self.reg, self.iterations)
        ok = (~unstable).to(normed.dtype)

        beta = self.reg * torch.log(v + M_EPS)  # (B', S*S)
        dens = density.index_select(0, sel).flatten(1)
        detached = dens.detach()
        total = detached.sum(1).clamp_min(1e-8)
        grad = beta / total[:, None] - (detached * beta).sum(1, keepdim=True) / (total * total)[:, None]
        loss = ((dens * grad.detach()).sum(1) * ok).sum()
        ot_obj = ((b * beta).sum(1) * ok).sum()
        wd = (torch.einsum("bn,bnm,bm->b", u, kernel * cost, v) * ok).sum()
        return loss, wd, ot_obj
