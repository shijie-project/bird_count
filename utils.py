"""Project-wide utilities: seeding, checkpoint rotation, metric averaging, logging."""

import logging
import os
import random
import sys
from collections import deque

import numpy as np
import torch


logger = logging.getLogger(__name__)

PathLike = str | os.PathLike


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


def setup_logging(debug: bool = False) -> None:
    """
    Configure specific logging format for industrial monitoring.
    Includes process name/ID to debug multi-process issues.
    """
    level = logging.DEBUG if debug else logging.INFO
    log_fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | PID:%(process)-5d | %(name)-35s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_fmt,
        datefmt=date_fmt,
        stream=sys.stdout,
    )

    # Suppress noisy libraries
    for logger_name in ["urllib3", "PIL", "matplotlib", "socket"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_cuda() -> None:
    """
    Configures global CUDA and cuDNN settings for maximum throughput.
    Enables TF32 on Ampere+ GPUs (matmul + cuDNN) for a ~2x speedup on
    fp32 paths with negligible numerical impact for this workload.
    """
    if not torch.cuda.is_available():
        return

    # Enable cuDNN auto-tuner to find the best algorithms for the current hardware.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # TF32 — free perf on Ampere (SM 8.0) and newer; ignored on older arches.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
            logger.info("Removed old checkpoint: %s", path)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Failed to remove %s: %s", path, e)


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
            self.logger.info("%-*s | %s", key_w, str(k), v)
        self.logger.info(sep)

    def __getattr__(self, name: str):
        # Fallback when an attribute isn't defined on the wrapper itself
        # (info/warning/error/debug/critical/log/...).
        return getattr(self.logger, name)


# --- Density visualization (shared by test.py, runtime monitor, viz tooling) ---
#
# The colormap is tied to an *absolute* density scale, not to each frame's own
# peak: a given color always means the same number of birds per cell, so frames
# are comparable to each other and to the alert thresholds. (Per-image min-max
# normalization used to be the convention here; it made an almost-empty frame
# look exactly as hot as a pile-up, since both got their peak painted red.)
#
# Unit: birds per model output cell, i.e. per `DOWNSAMPLE_RATIO`×`DOWNSAMPLE_RATIO`
# (8×8) block of source pixels — the same unit the model is trained on, where a
# GT cell holding one bird's center is exactly 1.0. Values >= `HEATMAP_VMAX` clip
# to the top of the colormap (dark red).
#
# Tuning `HEATMAP_VMAX`: it has to match the range the model actually emits, or
# the colormap collapses (too high -> everything blue, too low -> everything red).
# Measured on the val split with `ckpts/best_ep0980_mae4.73_mse5.81.pth`: cells
# are 0 almost everywhere, the 99th percentile is ~0.07, and per-frame peaks land
# in 0.12-0.22 birds/cell — a single bird's mass spreads over ~6-7 cells, so even
# an isolated bird peaks well below 1.0. 0.12 uses the full colormap: bird cores
# run warm, crowded regions merge into broad red areas, and anything denser clips
# at red. Re-measure after retraining, or after changing DOWNSAMPLE_RATIO (the
# unit itself is per-cell, so a stride change rescales every value).
#
# Note what the color does and doesn't tell you: it is *local* density, so a lone
# bird still has a warm core. Crowding shows up as the *area* covered by warm
# color, and a real pile-up is the one thing that pushes cells past `HEATMAP_VMAX`.
#
# `density_to_heatmap` returns both the colored heatmap and a binary mask, so
# callers can composite via `addWeighted` + `copyTo` (or just keep the heatmap,
# for side-by-side viz).
HEATMAP_VMAX = 0.12
HEATMAP_MIN_DENSITY = 0.02


def density_to_heatmap(
    density,
    *,
    vmax: float = HEATMAP_VMAX,
    min_density: float = HEATMAP_MIN_DENSITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a density map to a BGR JET heatmap + uint8 mask at its native
    resolution.

    Absolute (fixed-scale) colorization: `min_density` maps to the bottom of the
    colormap and `vmax` to the top, both in birds per output cell, with anything
    above `vmax` clipped. The same color therefore means the same density in
    every frame, and comparing two frames' heatmaps is meaningful. The mask is
    255 where the density exceeds `min_density`, 0 otherwise — pair with
    `cv2.copyTo` to blend only the cells that actually carry birds.

    Args:
        density: 2D `(H, W)` or 3D `(1, H, W)` / `(H, W, 1)` array or tensor,
            in birds per output cell (upsampling with INTER_LINEAR preserves
            this unit, so it is fine to colorize an upsampled map).
        vmax: density mapped to the top of the colormap. Raise it if the scene
            saturates to red, lower it if real pile-ups still look cool.
        min_density: densities at or below this are masked out (left as the raw
            frame) instead of washing the image in blue.

    Returns:
        heat: `(H, W, 3)` uint8 BGR JET heatmap.
        mask: `(H, W)` uint8 — 255 above `min_density`, 0 elsewhere.
    """
    import cv2  # heavy; lazy-imported.

    if torch.is_tensor(density):
        density = density.detach().cpu().numpy()
    density = np.asarray(density)
    if density.ndim == 3:
        density = density.squeeze()
    if density.ndim != 2:
        raise ValueError(f"density must be 2D after squeeze, got shape {density.shape}")

    scale = 255.0 / max(float(vmax), 1e-12)
    norm_u8 = np.clip(density.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    # Threshold in u8 space so the mask matches exactly what was colorized.
    mask = (norm_u8 > int(min_density * scale)).view(np.uint8) * np.uint8(255)
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

    from datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float64).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float64).reshape(3, 1, 1)
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
