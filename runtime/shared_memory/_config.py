import logging
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


class BufferState(IntEnum):
    """Strict state machine for ring-buffer slots; stored in the `state` field of METADATA_DTYPE."""

    FREE = 0  # Slot is empty, Grabber can write.
    WRITING = 1  # Grabber is currently writing (lock).
    READY = 2  # Data is written, ready for inference.
    READING = 3  # Inference is reading this slot (lock).


# Padded to 32 bytes so no record straddles a 64-byte cache line and two records fit per line.
# Field layout (with align=True): 1 + 3 pad + 4 + 4 + 4 pad + 8 + 8 = 32 bytes.
METADATA_DTYPE = np.dtype(
    [
        ("state", np.uint8),
        ("stream_id", np.int32),
        ("buffer_idx", np.int32),
        ("frame_idx", np.int64),
        ("timestamp", np.float64),
    ],
    align=True,
)


# ======================================================================
# Block configs
# ======================================================================


@dataclass(frozen=True, slots=True)
class _BlockConfigBase:
    """
    Common fields and size accounting for every SHM block config.
    Concrete subclasses just override the `dtype` default.
    """

    name: str
    shape: tuple
    dtype: Any  # str (e.g. "uint8") or np.dtype (METADATA_DTYPE)

    @property
    def itemsize(self) -> int:
        d = self.dtype if isinstance(self.dtype, np.dtype) else np.dtype(self.dtype)
        return d.itemsize

    @property
    def size_bytes(self) -> int:
        return int(np.prod(self.shape) * self.itemsize) if self.shape else 0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass(frozen=True, slots=True)
class FrameBlockConfig(_BlockConfigBase):
    """Raw camera frames: [stream, buffer, H, W, C] uint8 BGR."""

    dtype: str = "uint8"


@dataclass(frozen=True, slots=True)
class MetadataBlockConfig(_BlockConfigBase):
    """Slot state + provenance per (stream, buffer). Cache-line aligned."""

    dtype: np.dtype = METADATA_DTYPE

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024.0


@dataclass(frozen=True, slots=True)
class DensityBlockConfig(_BlockConfigBase):
    """
    Model-output-resolution density maps (H/stride, W/stride), keyed by the
    same (stream_id, buffer_idx) as frames. InferenceProcess writes here;
    MonitorHandler reads to compose the overlay at display time. The frame
    block is never written by inference, so raw frames stay intact for the
    recorder.
    """

    dtype: str = "float16"


@dataclass(frozen=True, slots=True)
class PreviewBlockConfig(_BlockConfigBase):
    """
    Downscaled BGR copies of frames (H/stride, W/stride, C), keyed by the
    same (stream_id, buffer_idx). InferenceProcess writes here; MonitorHandler
    reads here instead of the full frame block so the dashboard can resize a
    much smaller source.
    """

    dtype: str = "uint8"


# ======================================================================
# Aggregate config
# ======================================================================


@dataclass(frozen=True, slots=True)
class SharedMemoryConfig:
    """
    Aggregate descriptor for every SHM block the pipeline uses.
    Picklable + immutable, so it can be shipped across spawn boundaries.

    Optional blocks (density, preview) are None when disabled, rather than
    empty strings / tuples. Callers should check `cfg.density is not None`
    or iterate via `active_blocks()`.
    """

    frames: FrameBlockConfig
    metadata: MetadataBlockConfig
    density: Optional[DensityBlockConfig] = None
    preview: Optional[PreviewBlockConfig] = None

    def field_names(self) -> list[str]:
        """All block-field names in declaration order, including disabled ones."""
        return [f.name for f in fields(self)]

    def active_blocks(self) -> dict[str, _BlockConfigBase]:
        """Field-name -> block config, excluding any disabled (None) blocks."""
        return {f.name: cfg for f in fields(self) if (cfg := getattr(self, f.name)) is not None}

    @property
    def total_size_mb(self) -> float:
        return sum(cfg.size_mb for cfg in self.active_blocks().values())


# ======================================================================
# Builder
# ======================================================================


def build_memory_config(
    name_prefix: str,
    num_streams: int,
    num_buffers: int,
    resolution: tuple,
    channels: int,
    dtype: str,
    density_stride: int,
    density_dtype: str,
    preview_stride: int,
    preview_dtype: str,
) -> SharedMemoryConfig:
    height, width = resolution

    # Frame block: N-B-H-W-C (optimal for BGR-to-RGB flip on the inference side).
    frames = FrameBlockConfig(
        name=f"{name_prefix}_frames",
        shape=(num_streams, num_buffers, height, width, channels),
        dtype=dtype,
    )

    # Metadata block: one record per (stream, buffer).
    metadata = MetadataBlockConfig(
        name=f"{name_prefix}_meta",
        shape=(num_streams, num_buffers),
    )

    # Density at model-output resolution. Same first two dims as `frames` so
    # the same (sid, b_idx) pair indexes both blocks.
    if height % density_stride or width % density_stride:
        raise ValueError(f"resolution {resolution} not divisible by density_stride {density_stride}")
    density = DensityBlockConfig(
        name=f"{name_prefix}_density",
        shape=(num_streams, num_buffers, height // density_stride, width // density_stride),
        dtype=density_dtype,
    )

    # Preview block (optional). Stride=4 on a 720x1080 input gives a
    # 180x270 BGR tile (~140 KB/slot). preview_stride=0 disables the block
    # entirely (saves RAM on headless deployments).
    preview: Optional[PreviewBlockConfig] = None
    if preview_stride > 0:
        if height % preview_stride or width % preview_stride:
            raise ValueError(f"resolution {resolution} not divisible by preview_stride {preview_stride}")
        preview = PreviewBlockConfig(
            name=f"{name_prefix}_preview",
            shape=(
                num_streams,
                num_buffers,
                height // preview_stride,
                width // preview_stride,
                channels,
            ),
            dtype=preview_dtype,
        )

    return SharedMemoryConfig(frames=frames, metadata=metadata, density=density, preview=preview)
