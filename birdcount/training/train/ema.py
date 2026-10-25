"""Exponential moving average of the model's weights: what gets validated and saved."""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    """`ema = decay * ema + (1 - decay) * model` after every optimizer step.

    Float tensors (parameters and BN statistics) are averaged with one fused
    `_foreach_lerp_` per step instead of one kernel per tensor (v2: ~350 small
    launches); integer buffers (BN's `num_batches_tracked`) are copied.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self._ema_float, self._ema_int = self._split(self.model)

    @staticmethod
    def _split(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        tensors = list(model.state_dict(keep_vars=False).values())
        return [t for t in tensors if t.dtype.is_floating_point], [t for t in tensors if not t.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src_float, src_int = self._split(model)
        torch._foreach_lerp_(self._ema_float, src_float, 1.0 - self.decay)
        for dst, src in zip(self._ema_int, src_int, strict=True):
            dst.copy_(src)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)
