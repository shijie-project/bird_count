"""The inference pipeline — the one definition of how a frame becomes a count.

Every consumer (live deployment, evaluation, video analysis) runs frames
through `DensityPredictor`, so they all count the same way:

    BGR uint8 frames at the model geometry (see `frames.fit_frame`)
      -> H2D from a pinned staging buffer
      -> region mask on the frame (uint8 multiply: ignored pixels become black)
      -> normalise to the model dtype in one fused op (fp16 on CUDA)
      -> model forward (BGR-ordered first conv, see `DensityNet.set_input_order`)
      -> region mask on the density map (area-weighted, see `masks`)
      -> count = density sum, centroid = mass-weighted centre, both in fp32
      -> one D2H for the stats, and optionally the density map

The frame mask is applied before normalisation on purpose: training saw the
ignored region as literal black pixels, and black normalises to -mean/std,
not 0.

Allocation-free in steady state: staging and result buffers are pinned and
reused, so a `Prediction`'s arrays are views valid until the next `run`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from birdcount.core.frames import DEFAULT_FRAME_SIZE, fit_frame
from birdcount.core.masks import RegionMasks, density_weights
from birdcount.core.model import OUTPUT_STRIDE, DensityNet


logger = logging.getLogger(__name__)

# ImageNet statistics the backbone was trained with, RGB order.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Below this much total density there is nothing to take the centre of (an
# empty pen, a 'NO SIGNAL' tile); the delivery package's cut-off.
MIN_CENTROID_MASS = 1e-6


@dataclass(frozen=True)
class Prediction:
    counts: np.ndarray  # (n,) float32
    centroids: np.ndarray  # (n, 2) float32 (x, y) in 0..1 of the frame, NaN where empty
    density: np.ndarray | None  # (n, H/8, W/8) float16, when requested


class DensityPredictor:
    """Runs batches of frames through the model.

    Two ways in:

    * `predict(frames)` — any number of frames of any size; fitted, batched
      and copied for you. For offline tools and evaluation.
    * write up to `max_batch` frames into `staging[:n]` yourself, then
      `run(n)` — no extra copy. For the live pipeline, which stages straight
      from shared memory. Results are views into reused buffers.

    `mask_ids[i]` picks the `RegionMasks` row for frame i; None runs unmasked.
    """

    def __init__(
        self,
        model: DensityNet,
        *,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        masks: RegionMasks | None = None,
        max_batch: int = 1,
    ):
        if model.input_order != "bgr":
            raise ValueError("DensityPredictor feeds BGR frames; load the model with input_order='bgr'")
        width, height = frame_size
        if width % OUTPUT_STRIDE or height % OUTPUT_STRIDE:
            raise ValueError(f"frame size {frame_size} must be a multiple of {OUTPUT_STRIDE}")
        if masks is not None and masks.keep.shape[1:] != (height, width):
            raise ValueError(f"masks are {masks.keep.shape[2]}x{masks.keep.shape[1]}, frames {width}x{height}")

        param = next(model.parameters())
        self.model = model
        self.device, self.dtype = param.device, param.dtype
        self.frame_size = frame_size
        self.max_batch = max_batch
        self.masks = masks
        self.density_hw = (height // OUTPUT_STRIDE, width // OUTPUT_STRIDE)
        cuda = self.device.type == "cuda"

        # (x/255 - mean)/std == x*scale + bias, in BGR order to match the model.
        mean = torch.tensor(IMAGENET_MEAN[::-1], dtype=torch.float64).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD[::-1], dtype=torch.float64).view(1, 3, 1, 1)
        self._scale = (1.0 / (255.0 * std)).to(self.device, self.dtype)
        self._bias = (-mean / std).to(self.device, self.dtype)

        self._keep = self._weights = None
        if masks is not None:
            self._keep = torch.from_numpy(masks.keep).to(self.device).unsqueeze(1)  # (S, 1, H, W) uint8
            weights = density_weights(masks.keep, OUTPUT_STRIDE)
            self._weights = torch.from_numpy(weights).to(self.device, self.dtype).unsqueeze(1)

        h, w = self.density_hw
        self._staging = torch.empty((max_batch, height, width, 3), dtype=torch.uint8, pin_memory=cuda)
        self._stats_out = torch.empty((max_batch, 3), dtype=torch.float32, pin_memory=cuda)
        self._density_out = torch.empty((max_batch, h, w), dtype=torch.float16, pin_memory=cuda)
        self.staging: np.ndarray = self._staging.numpy()  # (max_batch, H, W, 3) BGR uint8

        x = (torch.arange(w, dtype=torch.float32) + 0.5) / w
        y = (torch.arange(h, dtype=torch.float32) + 0.5) / h
        self._xs, self._ys = x.to(self.device), y.to(self.device)

    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run(self, n: int, mask_ids: Sequence[int] | None = None, *, want_density: bool = False) -> Prediction:
        """Predict `staging[:n]`."""
        if not 0 < n <= self.max_batch:
            raise ValueError(f"batch of {n} outside 1..{self.max_batch}")
        if mask_ids is not None:
            if self.masks is None:
                raise ValueError("mask_ids given but the predictor has no masks")
            if len(mask_ids) != n:
                raise ValueError(f"{len(mask_ids)} mask ids for {n} frames")
            ids = torch.as_tensor(mask_ids, dtype=torch.long).to(self.device, non_blocking=True)

        frames = self._staging[:n].to(self.device, non_blocking=True).permute(0, 3, 1, 2)  # NCHW view
        if mask_ids is not None:
            frames = frames * self._keep.index_select(0, ids)  # 0/1 uint8: ignored pixels -> black
        x = frames.to(self.dtype, memory_format=torch.contiguous_format)
        x = torch.addcmul(self._bias, x, self._scale, out=x)

        density = self.model(x)  # (n, 1, h, w)
        if mask_ids is not None:
            density.mul_(self._weights.index_select(0, ids))

        d = density[:, 0]
        rows = d.sum(dim=2, dtype=torch.float32)  # (n, h): mass per row
        cols = d.sum(dim=1, dtype=torch.float32)  # (n, w): mass per column
        mass = rows.sum(dim=1)
        cx = (cols * self._xs).sum(dim=1) / mass
        cy = (rows * self._ys).sum(dim=1) / mass
        empty = mass <= MIN_CENTROID_MASS
        stats = torch.stack([mass, cx.masked_fill(empty, float("nan")), cy.masked_fill(empty, float("nan"))], dim=1)

        self._stats_out[:n].copy_(stats, non_blocking=True)
        if want_density:
            self._density_out[:n].copy_(d, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

        stats_np = self._stats_out[:n].numpy()
        return Prediction(
            counts=stats_np[:, 0],
            centroids=stats_np[:, 1:],
            density=self._density_out[:n].numpy() if want_density else None,
        )

    def predict(
        self, frames: Sequence[np.ndarray], mask_ids: Sequence[int] | None = None, *, want_density: bool = False
    ) -> Prediction:
        """Predict any number of BGR uint8 frames; each is fitted to the model geometry.

        Returned arrays are owned by the caller (copied out of the reused buffers).
        """
        counts, centroids, densities = [], [], []
        for start in range(0, len(frames), self.max_batch):
            chunk = frames[start : start + self.max_batch]
            for i, frame in enumerate(chunk):
                self.staging[i] = fit_frame(frame, self.frame_size)
            ids = None if mask_ids is None else mask_ids[start : start + len(chunk)]
            p = self.run(len(chunk), ids, want_density=want_density)
            counts.append(p.counts.copy())
            centroids.append(p.centroids.copy())
            if want_density:
                densities.append(p.density.copy())
        h, w = self.density_hw
        return Prediction(
            counts=np.concatenate(counts) if counts else np.empty(0, np.float32),
            centroids=np.concatenate(centroids) if centroids else np.empty((0, 2), np.float32),
            density=(np.concatenate(densities) if densities else np.empty((0, h, w), np.float16))
            if want_density
            else None,
        )

    def warmup(self) -> None:
        """Run every batch size once so cuDNN autotuning never stalls a real batch."""
        self.staging[:] = 0
        for n in range(1, self.max_batch + 1):
            self.run(n, None if self.masks is None else [0] * n, want_density=True)
