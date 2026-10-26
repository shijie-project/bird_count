"""Sinkhorn-Knopp solver for entropic-regularized optimal transport.

Pared-down PyTorch port of `ot.bregman.sinkhorn` from POT
(https://pythonot.github.io). Only the basic Sinkhorn-Knopp variant is kept
since DM-Count uses small problems where log-stabilization is unnecessary.

Reference: M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal
Transport", NeurIPS 2013.
"""

import logging

import torch


logger = logging.getLogger(__name__)
M_EPS = 1e-16


def sinkhorn(a, b, C, reg=1e-1, max_iter=1000, stop_thr=1e-9, log=True):
    """Solve min_gamma <gamma, C> + reg * H(gamma) s.t. row/col marginals = a, b.

    Args:
        a: (na,) target marginal (sums to 1).
        b: (nb,) source marginal (sums to 1).
        C: (na, nb) cost matrix.
        reg: entropic regularization strength.
        max_iter: max Sinkhorn iterations.
        stop_thr: stop when (b - K^T u v) MSE < stop_thr.
        log: if True, return (P, dict) with `u`, `v`, `alpha`, `beta`, `err`.
            The dict additionally contains `instability=True` if iteration was
            rolled back due to NaN/Inf — callers should treat the returned
            dual potentials as suspect in that case.

    Returns:
        P: (na, nb) transport plan, or (P, log_dict) if `log=True`.
    """
    device = a.device
    na, nb = C.shape

    if na < 1 or nb < 1:
        raise ValueError(f"C must be non-empty, got shape {tuple(C.shape)}")
    if na != len(a) or nb != len(b):
        raise ValueError(f"shape of a ({len(a)}) / b ({len(b)}) must match C ({tuple(C.shape)})")
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")
    if a.min() < 0.0 or b.min() < 0.0:
        raise ValueError("a/b must be non-negative")

    log_dict = {"err": []} if log else None

    u = torch.full((na,), 1.0 / na, dtype=a.dtype, device=device)
    v = torch.full((nb,), 1.0 / nb, dtype=b.dtype, device=device)

    K = torch.exp(C / -reg)

    err = float("inf")
    it = 0
    instability = False
    while err > stop_thr and it < max_iter:
        upre, vpre = u, v
        v = b / (torch.matmul(u, K) + M_EPS)
        u = a / (torch.matmul(K, v) + M_EPS)

        if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
            logger.warning("sinkhorn: numerical instability at iter %d, rolling back", it)
            u, v = upre, vpre
            instability = True
            break

        if log and (it % 10 == 0):
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            log_dict["err"].append(err)

        it += 1

    if log:
        log_dict["u"] = u
        log_dict["v"] = v
        log_dict["alpha"] = reg * torch.log(u + M_EPS)
        log_dict["beta"] = reg * torch.log(v + M_EPS)
        if instability:
            log_dict["instability"] = True

    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return (P, log_dict) if log else P
