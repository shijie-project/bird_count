"""Base classes shared by every result handler.

`BaseHandler` covers the dispatch contract (handle / handle_batch) and audit-log
lifecycle. `GUIToggleMixin` adds the flip helpers (`toggle`, `toggle_stream`)
for handlers that expose a GUI surface — they only need to implement
`enable / disable / is_enabled` (and the per-stream trio if applicable) and the
mixin fills in the toggling for free.

Handlers that need a GUI surface structurally satisfy the Protocols defined in
`runtime.gui._base` (MonitorTogglable, RecorderTogglable) — no explicit inheritance
required on the GUI side.

Note on `BaseHandler` vs `ABC`: we inherit from `ABC` but declare no
`@abstractmethod`, because the real contract is "override **at least one** of
`handle()` / `handle_batch()`" — a constraint `@abstractmethod` cannot express.
The runtime `NotImplementedError` in `handle()` enforces it.
"""

import logging
from abc import ABC
from typing import Optional, Union

import numpy as np

from runtime.audit import AuditLog
from runtime.config import Config
from runtime.inferencer import BatchInferenceResult, InferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig


logger = logging.getLogger(__name__)


class _NullAudit:
    """No-op stand-in for `AuditLog` so `handler.audit.log(...)` is safe to call
    before `start()`, after `stop()`, or when audit logging is disabled."""

    def log(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class GUIToggleMixin:
    """Implements `toggle()` / `toggle_stream()` on top of enable/disable/is_enabled.

    Subclasses must implement `enable`, `disable`, `is_enabled` (for aggregate
    control) and optionally `enable_stream`, `disable_stream`, `is_stream_enabled`
    (for per-stream control).
    """

    def toggle(self) -> bool:
        if self.is_enabled():
            return self.disable()
        return self.enable()

    def toggle_stream(self, stream_id: int) -> bool:
        if self.is_stream_enabled(stream_id):
            return self.disable_stream(stream_id)
        return self.enable_stream(stream_id)


class BaseHandler(ABC):
    """Abstract base for result handlers.

    Override `handle()` for per-frame processing, or override `handle_batch()`
    directly for batch-oriented work (e.g., when forwarding to a worker process).
    """

    # Whether `handle_batch` should resolve SHM frames before calling `handle`.
    # Override on subclasses that need raw pixels in the handler process.
    needs_frames: bool = False

    def __init__(self, config: Config, shm_config: SharedMemoryConfig, name: Optional[str] = None):
        self.config = config
        self.shm_config = shm_config
        self.name = name or self.__class__.__name__

        # The real AuditLog is opened in start() (file handles don't survive
        # process spawn). Until then, route audit calls to a no-op stub so
        # self.audit.log(...) is safe before start(), after stop(), and when no
        # audit path is configured.
        self._audit_log_path = config.envs.audit_log_path
        self.audit: Union[AuditLog, _NullAudit] = _NullAudit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the audit log. Subclasses that spawn processes override and call super()."""
        self.audit = AuditLog(self._audit_log_path, name=self.name)
        self.audit.log("handler.start", handler=self.name)

    def stop(self) -> None:
        """Close the audit log. Subclasses that own processes override and call super()."""
        self.audit.log("handler.stop", handler=self.name)
        self.audit.close()
        self.audit = _NullAudit()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemory) -> set[tuple[int, int]]:
        """Default: loop over the batch calling `handle` per result.

        Returns a set of (sid, b_idx) pairs the handler is asynchronously
        claiming — ResultProcess defers SHM release for those until the
        handler acks via `ack_queue`. Default = empty set (synchronous).
        """
        results = batch_result.results
        if not results:
            return set()

        frames = shm_client.frames if self.needs_frames else None
        for result in results:
            try:
                frame = frames[result.stream_id, result.buffer_idx] if frames is not None else None
                self.handle(result, frame)
            except Exception as e:
                logger.error("[%s] Error handling frame: %s", self.name, e, exc_info=True)
        return set()

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        """Process a single result. Override when you don't override `handle_batch`."""
        raise NotImplementedError(f"{type(self).__name__} must override handle() or handle_batch().")

    # ------------------------------------------------------------------
    # Optional surface — handlers that don't expose these get no-op defaults
    # ------------------------------------------------------------------

    def get_active_devices(self) -> set[str]:
        """Devices currently in ACTIVE state. Override per handler."""
        return set()

    def cancel_all(self) -> None:
        """Drop in-flight alert lifecycles. Default no-op; override per handler."""
