import logging
from dataclasses import dataclass, field
from typing import Optional

import torch


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    """DTO for a single frame result. Note: frozen=False for flexible post-processing.

    The inferencer fills the fields up to `centroid_y`. `centroid_x/y` is the
    mass-weighted centre of the (masked) density map in normalized image
    coordinates, 0..1 across the frame — what the alarm's directional-motion
    filter tracks — and None when the map holds no density. The rest are
    written in place by the consumer's alarm engine before any handler sees
    the result:

        alert_flag       an alarm event is open on this stream, or it is hijacked
        manual_override  the debug GUI's manual hijack drives this stream's
                         devices directly (no alarm event)
        simulated        the hijack is instead simulating a pile-up through the
                         alarm (DEBUG=1 + SMS dry-run, see _AlarmEngine)
        alarm_level      level of the open event (0 = none)
        alarm_actions    the `alarm.AlarmAction`s this sample produced
                         (level1/2/3, recovery) — what handlers react to
        alarm_sample     the `alarm.CountSample` the decision was made on
        alarm_event_id   id of the event open after this sample, if any
    """

    stream_id: int
    buffer_idx: int
    frame_idx: int
    timestamp: float
    count: float
    latency: float
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None
    alert_flag: bool = False
    manual_override: bool = False
    simulated: bool = False
    alarm_level: int = 0
    alarm_actions: tuple = ()
    alarm_sample: Optional[object] = None
    alarm_event_id: Optional[str] = None


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


# Below this much total density there is nothing to take the centre of (an
# empty pen, a 'NO SIGNAL' tile); same cut-off as the delivery package.
_MIN_CENTROID_MASS = 1e-6


def density_centroids(density_map: torch.Tensor) -> torch.Tensor:
    """Mass-weighted centre of each density map, normalized to 0..1.

    `density_map` is (B, 1, H, W) or (B, H, W). Returns (B, 2) float32 of
    `(x, y)` as `(index + 0.5) / size` — identical to the delivery package's
    `model_runtime/inference.py:density_centroid` — with NaN where the map's
    mass is below `_MIN_CENTROID_MASS`. Stays on the input's device.
    """
    d = density_map.float()
    if d.dim() == 4:
        d = d[:, 0]
    _, h, w = d.shape
    xs = (torch.arange(w, device=d.device, dtype=torch.float32) + 0.5) / w
    ys = (torch.arange(h, device=d.device, dtype=torch.float32) + 0.5) / h
    mass = d.sum(dim=(1, 2))
    cx = (d.sum(dim=1) * xs).sum(dim=1) / mass
    cy = (d.sum(dim=2) * ys).sum(dim=1) / mass
    out = torch.stack([cx, cy], dim=1)
    out[mass <= _MIN_CENTROID_MASS] = float("nan")
    return out


def permute_first_conv_for_bgr(model: torch.nn.Module, name: str = "InferenceWorker") -> bool:
    """Swap the input-channel order (R↔B) of the model's first 3-channel Conv2d.

    Why: cv2 / cameras deliver frames in BGR. The model was trained on RGB
    (PIL/`Image.convert("RGB")` in datasets/bird.py). The mathematically
    equivalent way to consume BGR without flipping the tensor at inference
    time is to permute the first conv layer's input-channel weights once.

    Net effect: drops a 50MB GPU allocation + one kernel per batch, with no
    numerical change in the model's output. The caller MUST also feed mean/std
    in BGR order; otherwise normalization is applied to the wrong channels.

    Not idempotent — calling this twice on the same model un-permutes it.
    Call exactly once during init.
    """
    conv1 = getattr(getattr(model, "backbone", None), "conv1", None)
    first_conv = None
    if isinstance(conv1, torch.nn.Conv2d):
        first_conv = conv1
    elif conv1 is not None:
        first_conv = next((m for m in conv1.modules() if isinstance(m, torch.nn.Conv2d)), None)
    if first_conv is None or first_conv.weight.shape[1] != 3:
        logger.warning(
            "[%s] Could not locate first Conv2d(in_channels=3); BGR direct-feed disabled. "
            "Inference will still work but pays a flip per batch.",
            name,
        )
        return False
    with torch.no_grad():
        first_conv.weight.copy_(first_conv.weight[:, [2, 1, 0], :, :])
    logger.info("[%s] Permuted conv1 input channels for BGR direct-feed.", name)
    return True


def get_optimal_memory_format(device: torch.device) -> torch.memory_format:
    """Auto-detects the optimal memory layout based on GPU architecture."""
    if device.type != "cuda":
        return torch.contiguous_format
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception as e:
        logger.warning("Failed to detect GPU capability: %s. Fallback to NCHW.", e)
        return torch.contiguous_format

    if (major, minor) >= (7, 5):
        logger.info("Auto-Tuning: Arch %d.%d detected. Using 'channels_last' (NHWC).", major, minor)
        return torch.channels_last
    logger.info("Auto-Tuning: Arch %d.%d detected. Using 'contiguous_format' (NCHW).", major, minor)
    return torch.contiguous_format
