"""Colour augmentation and normalisation of a training batch, on the GPU.

v2 ran torchvision's PIL ColorJitter per sample in the DataLoader workers —
the single most expensive step of its data pipeline (~9 ms per sample). Here
the same operations run batched on the GPU, after the uint8 crops arrive:

    BGR uint8 -> RGB float [0, 1]
    -> brightness, contrast, saturation, hue   (one random order per batch,
                                                 independent factors per sample)
    -> gamma (with probability p)
    -> ImageNet normalisation
    -> Gaussian noise in normalised space (with probability p)

The colour operations follow torchvision's tensor definitions
(`transforms._functional_tensor`, BSD-3-Clause) so the factors mean the same
thing; tests compare against torchvision directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from birdcount.core.inference import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class PhotometricConfig:
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.3
    hue: float = 0.1
    gamma_range: tuple[float, float] = (0.7, 1.3)
    gamma_p: float = 0.5
    noise_std: float = 0.02
    noise_p: float = 0.5


def _grayscale(x: Tensor) -> Tensor:
    r, g, b = x.unbind(dim=1)
    return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(1)


def adjust_brightness(x: Tensor, f: Tensor) -> Tensor:
    return (x * f).clamp_(0, 1)


def adjust_contrast(x: Tensor, f: Tensor) -> Tensor:
    mean = _grayscale(x).mean(dim=(1, 2, 3), keepdim=True)
    return (f * x + (1 - f) * mean).clamp_(0, 1)


def adjust_saturation(x: Tensor, f: Tensor) -> Tensor:
    return (f * x + (1 - f) * _grayscale(x)).clamp_(0, 1)


def adjust_hue(x: Tensor, shift: Tensor) -> Tensor:
    """Rotate hue by `shift` (fraction of a turn, in [-0.5, 0.5])."""
    r, g, b = x.unbind(dim=1)
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    crd = torch.where(eqc, ones, cr)
    rc, gc, bc = (maxc - r) / crd, (maxc - g) / crd, (maxc - b) / crd
    h = (maxc == r) * (bc - gc) + ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    h = h + ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = torch.fmod(h / 6.0 + 1.0, 1.0)

    h = (h + shift[:, 0]) % 1.0
    v = maxc
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(torch.int64) % 6
    p = (v * (1.0 - s)).clamp(0, 1)
    q = (v * (1.0 - s * f)).clamp(0, 1)
    t = (v * (1.0 - s * (1.0 - f))).clamp(0, 1)
    # Sextant i -> (r, g, b), torchvision's table:
    #   i: 0 1 2 3 4 5
    #   r: v q p p t v
    #   g: t v v q p p
    #   b: p p t v v q
    # Nested `where` instead of torchvision's 18-plane stack + einsum: same
    # values at a fraction of the memory.
    r = torch.where((i == 0) | (i == 5), v, torch.where(i == 1, q, torch.where(i == 4, t, p)))
    g = torch.where((i == 1) | (i == 2), v, torch.where(i == 0, t, torch.where(i == 3, q, p)))
    b = torch.where((i == 3) | (i == 4), v, torch.where(i == 2, t, torch.where(i == 5, q, p)))
    return torch.stack((r, g, b), dim=1)


class Photometric:
    """Turns a (B, H, W, 3) BGR uint8 batch on the device into normalised RGB floats."""

    def __init__(self, config: PhotometricConfig | None = None):
        self.config = config or PhotometricConfig()

    def __call__(self, images: Tensor, *, augment: bool, generator: torch.Generator | None = None) -> Tensor:
        """`generator` (CPU) makes the random factors reproducible."""
        device = images.device
        x = images.permute(0, 3, 1, 2).flip(1).float().div_(255.0)  # BGR -> RGB
        n = x.shape[0]
        cfg = self.config

        def uniform(lo: float, hi: float) -> Tensor:
            u = torch.rand(n, generator=generator) * (hi - lo) + lo
            return u.view(n, 1, 1, 1).to(device)

        def chosen(p: float) -> Tensor:
            return (torch.rand(n, generator=generator) < p).view(n, 1, 1, 1).to(device)

        if augment:
            ops = {
                0: lambda t: adjust_brightness(t, uniform(max(0.0, 1 - cfg.brightness), 1 + cfg.brightness)),
                1: lambda t: adjust_contrast(t, uniform(max(0.0, 1 - cfg.contrast), 1 + cfg.contrast)),
                2: lambda t: adjust_saturation(t, uniform(max(0.0, 1 - cfg.saturation), 1 + cfg.saturation)),
                3: lambda t: adjust_hue(t, uniform(-cfg.hue, cfg.hue)),
            }
            for op in torch.randperm(4, generator=generator).tolist():
                x = ops[op](x)
            gamma = torch.where(chosen(cfg.gamma_p), uniform(*cfg.gamma_range), torch.ones((), device=device))
            x = x.pow_(gamma).clamp_(0, 1)

        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std

        if augment and cfg.noise_std > 0:
            noise_seed = int(torch.randint(0, 2**62, (1,), generator=generator))
            g = torch.Generator(device=device).manual_seed(noise_seed)
            noise = torch.randn(x.shape, generator=g, device=device) * cfg.noise_std
            x = x + noise * chosen(cfg.noise_p)
        return x
