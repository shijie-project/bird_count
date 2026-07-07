"""MonitorHandler — owns the DisplayProcess lifecycle and forwards batches to it."""

import logging
import queue
from multiprocessing import Process, Queue, Value
from typing import Optional

import numpy as np

from runtime.config import Config
from runtime.handlers import BaseHandler, GUIToggleMixin
from runtime.inferencer import BatchInferenceResult, InferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._process import DisplayProcess


logger = logging.getLogger(__name__)


# Bounded so a slow display process can't backlog batches indefinitely. Five
# batches is roughly 1/4 second at 20 FPS — enough to absorb a brief stall.
DISPLAY_QUEUE_MAXSIZE = 5

# Polite-shutdown budget for DisplayProcess before we kill -9 it.
PROC_TERMINATE_GRACE = 1.0
PROC_KILL_GRACE = 0.5


class MonitorHandler(GUIToggleMixin, BaseHandler):
    """Proxy handler that runs the dashboard UI in a dedicated process.

    Aggregate GUI surface only (no per-stream control). On `enable()` spawns a
    `DisplayProcess`, on `disable()` tears it down. `handle_batch` forwards
    each batch over `display_queue` and claims the buffer pairs so SHM stays
    pinned until DisplayProcess acks via `ack_queue`.

    Initial enable state is taken from `config.envs.enable_monitor`. After
    `start()`, the GUI owns the state.
    """

    # The handler itself never reads SHM — DisplayProcess does, off-process,
    # using its own SHM client. Setting needs_frames=False skips the unnecessary
    # frame indexing in BaseHandler.handle_batch (which we override anyway).
    needs_frames = False

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        ack_queue: Optional[Queue] = None,
        name: str = "Monitor",
    ):
        super().__init__(config=config, shm_config=shm_config, name=name)
        self.ack_queue = ack_queue

        self.display_queue: Optional[Queue] = None
        self.proc: Optional[Process] = None

        self._enabled = False
        self._started = False
        self._initial_enabled = bool(getattr(config.envs, "enable_monitor", False))

        # Lock-free density-map toggle shared with the DisplayProcess. Backed
        # by mp.Value so the renderer can poll it cheaply per tick.
        self._show_density_map = Value("b", False)
        if config.envs.show_density_map:
            self._show_density_map.value = True

    # ------------------------------------------------------------------
    # GUI surface  (matches runtime.gui._base.MonitorTogglable)
    # ------------------------------------------------------------------

    def enable(self) -> bool:
        if self._enabled:
            return True
        self._enabled = True
        # Defer the actual spawn until start() has connected SHM, otherwise
        # DisplayProcess would race the SHM ring's initialization.
        if self._started:
            self._spawn_display()
        self.audit.log("monitor.enable")
        logger.info("[%s] Monitor turned ON.", self.name)
        return True

    def disable(self) -> bool:
        if not self._enabled:
            return False
        self._enabled = False
        self._terminate_display()
        self.audit.log("monitor.disable")
        logger.info("[%s] Monitor turned OFF.", self.name)
        return False

    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Density-map overlay surface  (matches runtime.gui._base.DensityMapTogglable)
    # ------------------------------------------------------------------

    def toggle_density_map(self) -> bool:
        new_state = not bool(self._show_density_map.value)
        self._show_density_map.value = new_state
        self.audit.log("monitor.density_map", enabled=new_state)
        logger.info("[%s] Density Map overlay turned %s.", self.name, "ON" if new_state else "OFF")
        return new_state

    def is_density_map_enabled(self) -> bool:
        return bool(self._show_density_map.value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()
        self._started = True
        if self._initial_enabled:
            # Honor config.envs.enable_monitor — go straight through enable()
            # so the audit log + spawn happen exactly once.
            self.enable()

    def stop(self) -> None:
        self._terminate_display()
        super().stop()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        # Unused — handle_batch is overridden. Kept satisfied for clarity only.
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemory) -> set[tuple[int, int]]:
        """Forward the batch to DisplayProcess; claim its buffers on success."""
        if not self._enabled or self.display_queue is None or not batch_result.results:
            return set()
        try:
            self.display_queue.put_nowait(batch_result)
        except queue.Full:
            # Queue is bounded; dropping the batch is preferable to blocking
            # ResultProcess. Caller will release SHM immediately for these pairs.
            return set()
        return set(zip(batch_result.stream_ids, batch_result.buffer_indices))

    # ------------------------------------------------------------------
    # Display process plumbing
    # ------------------------------------------------------------------

    def _spawn_display(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        self.display_queue = Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
        self.proc = DisplayProcess(
            self.config,
            self.shm_config,
            self.display_queue,
            self.ack_queue,
            show_density_map=self._show_density_map,
        )
        self.proc.start()
        logger.info("[%s] Background display process started.", self.name)

    def _terminate_display(self) -> None:
        proc = self.proc
        self.proc = None
        self.display_queue = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.join(timeout=PROC_TERMINATE_GRACE)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=PROC_KILL_GRACE)
            logger.info("[%s] Background display process stopped.", self.name)
        except Exception as e:
            logger.error("[%s] Error terminating display: %s", self.name, e, exc_info=True)
