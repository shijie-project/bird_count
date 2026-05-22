import logging
import multiprocessing.shared_memory as shm
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ._config import (
    METADATA_DTYPE,
    BufferState,
    SharedMemoryConfig,
    build_memory_config,
)


logger = logging.getLogger(__name__)


@dataclass
class _Block:
    """
    Owns one POSIX shared-memory segment across its whole lifecycle.

    Master side: `allocate()` -> `cleanup()`
    Worker side: `attach()`   -> `detach()`

    The owner-side and worker-side method pairs are intentionally distinct so
    you can read a call site and instantly see which side is talking.
    """

    name: str
    shape: tuple
    dtype: Any  # str (e.g. "uint8") or np.dtype (METADATA_DTYPE)
    _shm: Optional[shm.SharedMemory] = field(default=None, repr=False)
    array: Optional[np.ndarray] = field(default=None, repr=False)
    stream_views: list[np.ndarray] = field(default_factory=list, repr=False)

    @classmethod
    def from_config(cls, cfg) -> "_Block":
        """Build a runtime block from any *BlockConfig (duck-typed: name/shape/dtype)."""
        return cls(name=cfg.name, shape=cfg.shape, dtype=cfg.dtype)

    @property
    def size_bytes(self) -> int:
        if not self.shape:
            return 0
        if isinstance(self.dtype, np.dtype):
            itemsize = self.dtype.itemsize
        else:
            itemsize = np.dtype(self.dtype).itemsize
        return int(np.prod(self.shape) * itemsize)

    # ------------------------------------------------------------------
    # Master side
    # ------------------------------------------------------------------

    def allocate(self):
        """Create the segment, or attach if one already exists with sufficient size."""
        size = self.size_bytes
        try:
            self._shm = shm.SharedMemory(name=self.name, create=True, size=size)
        except FileExistsError:
            logger.warning(f"'{self.name}' already exists. Attaching to existing block.")
            self._shm = shm.SharedMemory(name=self.name, create=False)
            if self._shm.size < size:
                raise ValueError(f"Existing shm '{self.name}' size ({self._shm.size}) < required ({size})")

    def cleanup(self):
        """Close and unlink. Idempotent."""
        if self._shm is None:
            return
        try:
            self._shm.close()
            self._shm.unlink()
            logger.info(f"'{self.name}' unlinked.")
        except Exception as e:
            logger.error(f"Error cleaning '{self.name}': {e}")
        finally:
            self._shm = None

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------

    def attach(self):
        """Attach to the segment by name and build the numpy view + per-stream cache."""
        self._shm = shm.SharedMemory(name=self.name)
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf, order="C")
        # shape[0] is num_streams for every block in this pipeline.
        self.stream_views = [self.array[i] for i in range(self.shape[0])]

    def detach(self):
        """Close the local handle and clear the numpy view. Idempotent."""
        if self._shm is None:
            return
        try:
            self._shm.close()
        except Exception as e:
            logger.error(f"Error detaching '{self.name}': {e}")
        finally:
            self._shm = None
            self.array = None
            self.stream_views = []

    @property
    def buf(self) -> Optional[memoryview]:
        return self._shm.buf if self._shm is not None else None


class SharedMemory:
    """
    Unified owner / accessor for the pipeline's SHM segments.

    Lifecycle:
      Master process:  allocate() -> (workers run) -> cleanup()
      Worker process:  connect()  -> use frames/metadata/... -> disconnect()

    Master and worker each construct their own instance from the same
    `SharedMemoryConfig`. Only the instance that called `allocate()` will
    `unlink()` on cleanup — workers that never allocated are a no-op,
    so a stray `cleanup()` call in a worker can't yank the segments
    out from under the rest of the pipeline.
    """

    def __init__(self, config: SharedMemoryConfig, name: str = "SharedMemory"):
        self.name = name
        self.config = config

        # One _Block per non-None field in the config. The field name doubles
        # as both the dict key here and the public attribute name on this
        # class after connect() (config.frames -> self._blocks["frames"]
        # -> self.frames). Adding a new block is now config-only: declare
        # the new field on SharedMemoryConfig and this dict picks it up
        # automatically via active_blocks().
        self._blocks: dict[str, _Block] = {
            name: _Block.from_config(cfg) for name, cfg in config.active_blocks().items()
        }

        # Ownership flag — set True by allocate(). Guards cleanup() against
        # accidental destructive calls from worker-side instances.
        self._owns = False

        # Worker-side public surface — populated by connect(), cleared by
        # disconnect(). Bound here as plain attributes so hot-loop access
        # stays a single attribute lookup (no property indirection).
        self.frames: Optional[np.ndarray] = None
        self.metadata: Optional[np.ndarray] = None
        self.density: Optional[np.ndarray] = None
        self.preview: Optional[np.ndarray] = None
        self.stream_frames: list[np.ndarray] = []
        self.stream_metadata: list[np.ndarray] = []
        self.stream_density: list[np.ndarray] = []
        self.stream_preview: list[np.ndarray] = []

    # ==================================================================
    # Master side — called once in the parent process
    # ==================================================================

    def allocate(self):
        """Create every SHM segment and initialize the metadata block."""
        try:
            for block in self._blocks.values():
                block.allocate()
            self._initialize_metadata()
            self._owns = True
            self._log_summary()
        except Exception as e:
            logger.error(f"[{self.name}] Allocation failed: {e}")
            self._cleanup_internal()  # tear down anything we did manage to allocate
            raise

    def cleanup(self):
        """
        Close and unlink every segment.
        No-op when called on a worker-side instance (`_owns == False`), so a
        stray `cleanup()` in a worker process can't destroy live segments.
        """
        if not self._owns:
            return
        self._cleanup_internal()
        self._owns = False

    def _cleanup_internal(self):
        for block in self._blocks.values():
            block.cleanup()

    def _initialize_metadata(self):
        """Vectorized initialization of the metadata block (master-side only)."""
        meta_array = np.ndarray(
            self.config.metadata.shape,
            dtype=METADATA_DTYPE,
            buffer=self._blocks["metadata"].buf,
        )
        meta_array["state"] = BufferState.FREE
        meta_array["stream_id"] = -1
        meta_array["buffer_idx"] = -1
        meta_array["frame_idx"] = 0
        meta_array["timestamp"] = 0.0

    def _log_summary(self):
        cfg = self.config
        parts: list[str] = []
        for name, block_cfg in cfg.active_blocks().items():
            # Blocks that report size_kb (just metadata today) are shown in KB;
            # everything else in MB.
            size_kb = getattr(block_cfg, "size_kb", None)
            if size_kb is not None:
                parts.append(f"{name.title()}: {size_kb:.2f} KB")
            else:
                parts.append(f"{name.title()}: {block_cfg.size_mb:.2f} MB")
        parts.append(f"Total: {cfg.total_size_mb:.2f} MB")
        parts.append(f"Shape: {cfg.frames.shape}")
        logger.info(f"[{self.name}] " + " | ".join(parts))

    # ==================================================================
    # Worker side — called once per worker process
    # ==================================================================

    def connect(self):
        """Attach to every segment and publish numpy views + per-stream caches."""
        try:
            for block in self._blocks.values():
                block.attach()
            # Publish each block's array and per-stream views under the field name
            # from the config (e.g. "frames" -> self.frames, self.stream_frames).
            for name, block in self._blocks.items():
                setattr(self, name, block.array)
                setattr(self, f"stream_{name}", block.stream_views)
            logger.debug(f"[{self.name}] Connected to {self.config.frames.name}")
        except Exception as e:
            logger.critical(f"[{self.name}] Connection failed: {e}")
            self.disconnect()
            raise

    def disconnect(self):
        """Detach from every segment and clear the public surface."""
        for block in self._blocks.values():
            block.detach()
        # Clear every public surface slot — including optional blocks that may
        # have been absent from self._blocks but were declared on the config.
        for name in self.config.field_names():
            setattr(self, name, None)
            setattr(self, f"stream_{name}", [])
        logger.debug(f"[{self.name}] Disconnected.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ==================================================================
    # Factory — build from raw sizing params instead of a pre-made config
    # ==================================================================

    @classmethod
    def build(
        cls,
        name_prefix: str,
        num_streams: int,
        num_buffers: int,
        resolution: tuple,
        channels: int = 3,
        dtype: str = "uint8",
        density_stride: int = 8,
        density_dtype: str = "float16",
        preview_dtype: str = "uint8",
        name: str = "SharedMemory",
    ) -> "SharedMemory":
        """
        Convenience factory: build the nested `SharedMemoryConfig` from raw
        sizing parameters and return an un-allocated instance. Call `.allocate()`
        next to actually create the segments.
        """
        config = build_memory_config(
            name_prefix=name_prefix,
            num_streams=num_streams,
            num_buffers=num_buffers,
            resolution=resolution,
            channels=channels,
            dtype=dtype,
            density_stride=density_stride,
            density_dtype=density_dtype,
            preview_stride=density_stride,
            preview_dtype=preview_dtype,
        )
        return cls(config, name=name)
