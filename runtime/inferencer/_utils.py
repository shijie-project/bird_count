import logging
from dataclasses import dataclass, field

import torch


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    """DTO for a single frame result. Note: frozen=False for flexible post-processing."""

    stream_id: int
    buffer_idx: int
    frame_idx: int
    timestamp: float
    count: float
    latency: float
    alert_flag: bool = False


@dataclass(slots=True)
class BatchInferenceResult:
    """Container for batch results. Optimized to reduce instantiation overhead."""

    results: list[InferenceResult] = field(default_factory=list)

    def append(self, result: InferenceResult):
        self.results.append(result)

    @property
    def stream_ids(self) -> list[int]:
        """Returns a list of stream IDs in the batch."""
        return [res.stream_id for res in self.results]

    @property
    def buffer_indices(self) -> list[int]:
        """Returns a list of SHM buffer indices in the batch."""
        return [res.buffer_idx for res in self.results]

    def __len__(self) -> int:
        return len(self.results)


def permute_first_conv_for_bgr(model: torch.nn.Module, name: str = "InferenceWorker") -> bool:
    """Swap the input-channel order (R↔B) of the model's first 3-channel Conv2d.

    Why: cv2 / cameras deliver frames in BGR. The model was trained on RGB
    (PIL/`Image.convert("RGB")` in datasets/bird.py). The mathematically
    equivalent way to consume BGR without flipping the tensor at inference
    time is to permute the first conv layer's input-channel weights once.

    Net effect: drops a 50MB GPU allocation + one kernel per batch, with no
    numerical change in the model's output. The caller MUST also feed mean/std
    in BGR order; otherwise normalization is applied to the wrong channels.
    """
    conv1 = getattr(getattr(model, "backbone", None), "conv1", None)
    first_conv = None
    if isinstance(conv1, torch.nn.Conv2d):
        first_conv = conv1
    elif conv1 is not None:
        first_conv = next((m for m in conv1.modules() if isinstance(m, torch.nn.Conv2d)), None)
    if first_conv is None or first_conv.weight.shape[1] != 3:
        logger.warning(
            f"[{name}] Could not locate first Conv2d(in_channels=3); BGR direct-feed disabled. "
            f"Inference will still work but pays a flip per batch."
        )
        return False
    with torch.no_grad():
        first_conv.weight.copy_(first_conv.weight[:, [2, 1, 0], :, :])
    logger.info(f"[{name}] Permuted conv1 input channels for BGR direct-feed.")
    return True


def get_optimal_memory_format(device: torch.device) -> torch.memory_format:
    """Auto-detects the optimal memory layout based on GPU architecture."""
    if device.type != "cuda":
        return torch.contiguous_format
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception as e:
        logger.warning(f"Failed to detect GPU capability: {e}. Fallback to NCHW.")
        return torch.contiguous_format

    if (major, minor) >= (7, 5):
        logger.info(f"Auto-Tuning: Arch {major}.{minor} detected. Using 'channels_last' (NHWC).")
        return torch.channels_last
    logger.info(f"Auto-Tuning: Arch {major}.{minor} detected. Using 'contiguous_format' (NCHW).")
    return torch.contiguous_format
