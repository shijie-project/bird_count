"""TaskDispatcher — orchestrates worker processes and runs the consumer loop in-thread.

After the latest refactor the dispatcher does two things:
  1. Spawns + supervises the worker processes (grabber + N inferencers).
  2. Hosts the `ResultConsumer` directly — its `tick()` is called from the
     dispatcher's main loop, alongside the periodic supervisor check.

This means the dispatcher runs on the main thread of the main process, which
is exactly where Tk wants to live. One fewer mp.Process, one fewer
result-queue IPC hop, and shutdown ordering becomes linear.
"""

import logging
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from .config import Config
from .handlers import init_handlers
from .inferencer import InferencerProcess
from .result_comsumer import ResultConsumer
from .shared_memory import SharedMemory
from .stream_grabber import GrabberProcess
from .utils import setup_logging


logger = logging.getLogger(__name__)


# Tunables in one place.
_SUPERVISOR_INTERVAL = 0.5  # seconds; minimum gap between supervisor sweeps
_PROC_JOIN_TIMEOUT = 2.0  # graceful shutdown deadline per process
_RESTART_JOIN_TIMEOUT = 0.5  # reap of a dead process before re-spawn
_TERMINATE_JOIN_TIMEOUT = 0.5  # post-terminate() grace before kill()


@dataclass
class _ProcessSlot:
    """A managed slot in the dispatcher's process registry."""

    key: str
    label: str
    factory: Callable[[], mp.Process]
    proc: Optional[mp.Process] = None
    restart_count: int = 0

    def start(self) -> None:
        self.proc = self.factory()
        self.proc.start()

    def is_dead(self) -> bool:
        return self.proc is not None and not self.proc.is_alive()

    def restart(self) -> None:
        if self.proc is not None:
            self.proc.join(timeout=_RESTART_JOIN_TIMEOUT)
        self.proc = self.factory()
        self.proc.start()
        self.restart_count += 1

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.stop()

    def join_or_terminate(self) -> None:
        if self.proc is None:
            return
        self.proc.join(timeout=_PROC_JOIN_TIMEOUT)
        if not self.proc.is_alive():
            return
        logger.warning(f"Process {self.proc.name} failed to stop. Terminating...")
        self.proc.terminate()
        self.proc.join(timeout=_TERMINATE_JOIN_TIMEOUT)
        if self.proc.is_alive():
            self.proc.kill()


class TaskDispatcher:
    """Orchestrator for the video-analytics pipeline.

    Owns the worker processes (slots) + the in-process `ResultConsumer`.
    The main loop interleaves consumer ticks with a rate-limited supervisor
    sweep over the slots. A slot exhausting its restart budget — or the
    consumer's Terminate button — sets `shutdown_event` and the loop exits.
    """

    MAX_RESTARTS = 5

    def __init__(self, config: Config, name: str = "TaskDispatcher"):
        self.name = name
        self.config = config

        # --- Cross-process communication ---
        # Result queue is bounded for backpressure: sized for ~3s of inference
        # at target FPS so transient consumer stalls (audit fsync, GUI redraw)
        # don't force inference to drop frames.
        queue_size = self.config.envs.num_workers_per_gpu * 30
        self.result_queue = mp.Queue(maxsize=queue_size)
        # Ack queue stays unbounded — acks are tiny and only used by
        # MonitorHandler/DisplayProcess now.
        self.ack_queue = mp.Queue()

        # One warmup event per inference worker. Each worker sets its event
        # after GPU warmup; the consumer waits on all of them before mounting
        # the GUI so the operator's first click lands on a warm pipeline.
        num_workers = self.config.envs.num_workers_per_gpu
        self.warmup_events: list = [mp.Event() for _ in range(num_workers)]

        # Single shutdown signal. Set by:
        #   - the debug GUI's "Terminate Program" button (via consumer),
        #   - the supervisor when a slot exhausts its restart budget,
        #   - any KeyboardInterrupt that propagates into run().
        self.shutdown_event = mp.Event()

        # Allocated in run().
        self.shm: Optional[SharedMemory] = None
        self.shm_config = None
        self._slots: list[_ProcessSlot] = []
        self._consumer: Optional[ResultConsumer] = None

    # ==================================================================
    # Public API
    # ==================================================================

    def run(self) -> None:
        """Allocate SHM, spawn workers, then run the consumer + supervisor loop."""
        setup_logging(debug=self.config.envs.debug)
        try:
            self._init_resources()
            self._build_slots()
            self._start_all_slots()
            self._consumer = self._build_consumer()
            # The consumer needs its own SHM client (distinct from the master
            # `self.shm` so disconnect doesn't unlink the segments).
            with SharedMemory(self.shm_config, name="ConsumerSHM") as shm_client:
                self._consumer.setup(shm_client)
                try:
                    self._main_loop(shm_client)
                finally:
                    self._consumer.cleanup(shm_client)
        except KeyboardInterrupt:
            logger.info(f"[{self.name}] Interruption received (Ctrl+C).")
        except Exception as e:
            logger.critical(f"[{self.name}] Global dispatcher failure: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Strict resource reclamation. Idempotent — safe to call from any state."""
        logger.info(f"[{self.name}] Initiating graceful shutdown...")
        # Two passes: signal stop on every slot first, THEN join. They all wind
        # down concurrently instead of serializing behind each child's latency.
        for slot in self._slots:
            slot.stop()
        for slot in self._slots:
            slot.join_or_terminate()
        if self.shm is not None:
            self.shm.cleanup()
            self.shm = None

    # ==================================================================
    # Initialization
    # ==================================================================

    def _init_resources(self) -> None:
        logger.info(f"[{self.name}] Allocating Shared Memory...")
        shm = SharedMemory.build(
            name_prefix=self.config.shm.name_prefix,
            num_streams=self.config.num_streams,
            num_buffers=self.config.num_buffers,
            resolution=(self.config.shm.height, self.config.shm.width),
        )
        shm.allocate()
        self.shm = shm
        self.shm_config = shm.config

    def _build_slots(self) -> None:
        """One slot per child process. Consumer no longer has a slot — it runs in-thread."""
        logger.info(f"[{self.name}] Initializing system components...")
        num_workers = self.config.envs.num_workers_per_gpu

        # Declaration order is also start order. Inferencers first so they're
        # warming up while the grabber starts capturing.
        self._slots = [
            *(
                _ProcessSlot(
                    key=f"inferencer-{i}",
                    label=f"InferenceWorker-{i}",
                    factory=partial(self._build_inferencer, i, num_workers),
                )
                for i in range(num_workers)
            ),
            _ProcessSlot(key="grabber", label="GrabberProcess", factory=self._build_grabber),
        ]

    def _start_all_slots(self) -> None:
        for slot in self._slots:
            slot.start()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def _build_consumer(self) -> ResultConsumer:
        """In-process consumer + every registered handler."""
        consumer = ResultConsumer(
            config=self.config,
            shm_config=self.shm_config,
            result_queue=self.result_queue,
            ack_queue=self.ack_queue,
            warmup_events=self.warmup_events,
            shutdown_event=self.shutdown_event,
        )
        for handler in init_handlers(self.config, self.shm_config, ack_queue=self.ack_queue):
            consumer.register_handler(handler)
        return consumer

    def _build_inferencer(self, worker_id: int, total_workers: int) -> InferencerProcess:
        return InferencerProcess(
            config=self.config,
            shm_config=self.shm_config,
            result_queue=self.result_queue,
            worker_id=worker_id,
            total_workers=total_workers,
            warmup_event=self.warmup_events[worker_id],
        )

    def _build_grabber(self) -> GrabberProcess:
        return GrabberProcess(config=self.config, shm_config=self.shm_config)

    # ==================================================================
    # Main loop
    # ==================================================================

    def _main_loop(self, shm_client: SharedMemory) -> None:
        """Consumer tick + periodic supervisor sweep, until shutdown_event fires."""
        last_supervisor_check = time.time()
        while not self.shutdown_event.is_set():
            self._consumer.tick(shm_client)

            now = time.time()
            if now - last_supervisor_check >= _SUPERVISOR_INTERVAL:
                if not self._supervisor_sweep():
                    return
                last_supervisor_check = now

        logger.info(f"[{self.name}] Shutdown requested; exiting main loop.")

    def _supervisor_sweep(self) -> bool:
        """Restart any dead slot. Return False if any slot exhausts its budget."""
        for slot in self._slots:
            if slot.is_dead() and not self._try_restart(slot):
                return False
        return True

    def _try_restart(self, slot: _ProcessSlot) -> bool:
        """Restart a dead slot in place. Return False if its budget is exhausted."""
        if slot.restart_count >= self.MAX_RESTARTS:
            logger.critical(
                f"[{self.name}] {slot.label} died and has exhausted its restart budget "
                f"({slot.restart_count}/{self.MAX_RESTARTS}). Escalating to full shutdown."
            )
            self.shutdown_event.set()
            return False
        logger.warning(
            f"[{self.name}] {slot.label} died. Restart attempt {slot.restart_count + 1}/{self.MAX_RESTARTS}."
        )
        slot.restart()
        return True
