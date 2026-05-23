"""Project-wide utilities: seeding, checkpoint rotation, metric averaging, logging."""

import logging
import os
import random
import sys
from collections import deque
from typing import Union

import numpy as np
import torch


logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    cuDNN defaults to `benchmark=True` (faster, non-deterministic algorithm
    selection). Pass `deterministic=True` to swap to deterministic kernels at
    the cost of speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # covers single and multi-GPU

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


class SaveHandle:
    """Track a rolling window of up to `max_num` checkpoint files.

    Each `append(path)` records the path; once the window is full, the oldest
    file's path is popped and the file is deleted from disk.
    """

    def __init__(self, max_num: int):
        if max_num < 1:
            raise ValueError(f"max_num must be >= 1, got {max_num}")
        self.max_num = max_num
        self._queue: "deque[str]" = deque()

    def append(self, path: PathLike) -> None:
        self._queue.append(os.fspath(path))
        while len(self._queue) > self.max_num:
            self._remove_file(self._queue.popleft())

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        return iter(self._queue)

    @staticmethod
    def _remove_file(path: str) -> None:
        try:
            os.remove(path)
            logger.info(f"Removed old checkpoint: {path}")
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Failed to remove {path}: {e}")


class AverageMeter:
    """Streaming average of a scalar (or `.item()`-able tensor)."""

    __slots__ = ("val", "avg", "sum", "count")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        if hasattr(val, "item"):  # accept torch tensors transparently
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter(val={self.val:.4g}, avg={self.avg:.4g}, n={self.count})"


_DEFAULT_LOGGER_NAME = "PileUpProject"
_DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


class Logger:
    """Thin wrapper around `logging.getLogger` with file + console handlers.

    Idempotent: re-instantiating with the same logger name reuses the existing
    handlers instead of duplicating them. Forwards every method on the
    underlying `logging.Logger` (info/warning/error/debug/critical/log/...).
    """

    def __init__(self, log_file: PathLike, name: str = _DEFAULT_LOGGER_NAME):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            formatter = logging.Formatter(_DEFAULT_FORMAT)

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)

            fh = logging.FileHandler(os.fspath(log_file))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)

            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def print_config(self, config: dict) -> None:
        """Log a config dict as a readable two-column table."""
        rows = list(config.items())
        if not rows:
            self.logger.info("(empty config)")
            return
        key_w = max(len(str(k)) for k, _ in rows)
        sep = "-" * (key_w + 30)
        self.logger.info(sep)
        for k, v in rows:
            self.logger.info(f"{str(k):<{key_w}} | {v}")
        self.logger.info(sep)

    def __getattr__(self, name: str):
        # Fallback when an attribute isn't defined on the wrapper itself
        # (info/warning/error/debug/critical/log/...).
        return getattr(self.logger, name)


# --- Density visualization (shared by test.py, runtime monitor, viz tooling) ---
#
# All three stages use per-image min-max normalization + a vmax floor so the
# *spatial* structure is visible regardless of absolute density magnitude, and
# blank-ish images aren't noise-amplified. `density_to_heatmap` returns both
# the colored heatmap and a binary mask, so callers can composite via
# `addWeighted` + `copyTo` (or just keep the heatmap, for side-by-side viz).
#
# Constants are module-level so each call site reads them from one place. Tune
# `HEATMAP_VMAX_FLOOR` if your model emits very low peak densities (lower
# floor) or peaks well above 1.0 (raise floor / clip differently). The current
# floor of 1e-2 matches DM-Count-style outputs on stride-8 grids.
HEATMAP_VMAX_FLOOR = 1e-2
HEATMAP_REL_THRESHOLD = 0.10


def density_to_heatmap(
    density,
    *,
    vmax_floor: float = HEATMAP_VMAX_FLOOR,
    threshold: float = HEATMAP_REL_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a density map to a BGR JET heatmap + uint8 mask at its native
    resolution.

    Per-image min-max normalization: each density's peak maps to color 255,
    so spatial structure is visible regardless of absolute magnitude. The
    `vmax_floor` lower bound on the denominator stops blank/near-blank images
    from amplifying noise. The mask is 255 where the normalized value exceeds
    `threshold * 255`, 0 otherwise — pair with `cv2.copyTo` to blend only the
    "above-threshold" cells.

    Args:
        density: 2D `(H, W)` or 3D `(1, H, W)` / `(H, W, 1)` array or tensor.
        vmax_floor: lower bound on per-image peak used as the normalization
            denominator. Quiet images stay quiet instead of going full-color.
        threshold: fraction of per-image peak below which cells are masked
            out (0.10 = hide the bottom 10%).

    Returns:
        heat: `(H, W, 3)` uint8 BGR JET heatmap.
        mask: `(H, W)` uint8 — 255 above threshold, 0 elsewhere.
    """
    import cv2  # heavy; lazy-imported.

    if torch.is_tensor(density):
        density = density.detach().cpu().numpy()
    density = np.asarray(density)
    if density.ndim == 3:
        density = density.squeeze()
    if density.ndim != 2:
        raise ValueError(f"density must be 2D after squeeze, got shape {density.shape}")

    vmax = max(float(density.max()), vmax_floor)
    norm_u8 = np.clip(density.astype(np.float32) * (255.0 / vmax), 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    mask = (norm_u8 > int(threshold * 255)).view(np.uint8) * np.uint8(255)
    return heat, mask


def visualize_save(img_tensor, pred_density, gt_density, save_path: PathLike) -> None:
    """Save [image | GT density | pred density] side-by-side as a PNG.

    Args:
        img_tensor: (3, H, W) ImageNet-normalized RGB tensor.
        pred_density: (1, h', w') or (h', w') density map (tensor or array).
        gt_density:  (1, h', w') or (h', w') density map (tensor or array).
        save_path:   Output PNG path.
    """
    import cv2  # heavy; lazy-imported.

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.detach().cpu().numpy()
    img = ((img * std + mean) * 255.0).clip(0, 255).transpose(1, 2, 0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    gt_heat, _ = density_to_heatmap(gt_density)
    pred_heat, _ = density_to_heatmap(pred_density)
    vis_gt = cv2.resize(gt_heat, (w, h), interpolation=cv2.INTER_CUBIC)
    vis_pred = cv2.resize(pred_heat, (w, h), interpolation=cv2.INTER_CUBIC)

    sep = np.full((h, 10, 3), 255, dtype=np.uint8)
    final = np.hstack([img, sep, vis_gt, sep, vis_pred])

    gt_count = _scalar_sum(gt_density)
    pred_count = _scalar_sum(pred_density)
    cv2.putText(
        final, f"GT: {gt_count:.1f}", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        final,
        f"Pred: {pred_count:.1f}",
        (2 * w + 30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(os.fspath(save_path), final):
        raise OSError(f"cv2.imwrite failed for {save_path!r}")


def _scalar_sum(x) -> float:
    if torch.is_tensor(x):
        return float(x.sum().item())
    return float(np.asarray(x).sum())
