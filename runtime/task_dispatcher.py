import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from .config import Config
from .handlers import init_handlers
from .inferencer import InferencerProcess
from .result_comsumer import ResultProcess
from .shared_memory import SharedMemory
from .stream_grabber import GrabberProcess
from .utils import setup_logging


logger = logging.getLogger(__name__)


# Tunables in one place.
_SUPERVISOR_INTERVAL = 0.1  # seconds; supervisor loop heartbeat
_PROC_JOIN_TIMEOUT = 2.0  # graceful shutdown deadline per process
_RESTART_JOIN_TIMEOUT = 0.5  # reap of a dead process before re-spawn
_TERMINATE_JOIN_TIMEOUT = 0.5  # post-terminate() grace before kill()


@dataclass
class _ProcessSlot:
    """
    A managed slot in the dispatcher's process registry. Bundles one child
    process with the factory that knows how to rebuild it and the restart
    bookkeeping that decides when to stop trying.
    """

    key: str  # short id used in logs / restart accounting
    label: str  # human-readable name for log messages
    factory: Callable[[], mp.Process]
    proc: Optional[mp.Process] = None
    restart_count: int = 0

    def start(self):
        self.proc = self.factory()
        self.proc.start()

    def is_dead(self) -> bool:
        """True iff this slot has been started and its process has exited."""
        return self.proc is not None and not self.proc.is_alive()

    def restart(self):
        """Reap the dead process and replace it with a fresh one from the factory."""
        if self.proc is not None:
            self.proc.join(timeout=_RESTART_JOIN_TIMEOUT)
        self.proc = self.factory()
        self.proc.start()
        self.restart_count += 1

    def stop(self):
        """Send the cooperative stop signal (children all implement .stop())."""
        if self.proc is not None:
            self.proc.stop()

    def join_or_terminate(self):
        """Wait for graceful exit; escalate to terminate -> kill if it overruns."""
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
    """
    Orchestrator for the multi-process video-analytics pipeline.

    Each child process lives in a `_ProcessSlot` that knows how to rebuild
    itself. The supervisor polls every slot at `_SUPERVISOR_INTERVAL`; when
    a slot's process dies, the dispatcher attempts up to `MAX_RESTARTS`
    in-place restarts before setting `shutdown_event` and tearing the
    whole pipeline down.
    """

    MAX_RESTARTS = 5

    def __init__(self, config: Config, name: str = "TaskDispatcher"):
        self.name = name
        self.config = config

        # Communication queues.
        # Result queue is bounded for backpressure: sized for ~3s of inference at
        # the target FPS so transient ResultProcess stalls (audit fsync, GUI
        # redraw, etc.) don't force the inferencer to drop frames.
        # Ack queue is unbounded since acks are tiny and rare; ResultProcess
        # drains it every loop iteration.
        queue_size = self.config.envs.num_workers_per_gpu * 30
        self.result_queue = mp.Queue(maxsize=queue_size)
        self.ack_queue = mp.Queue()

        # One warmup event per inference worker. Each worker sets its event
        # after GPU warmup; ResultProcess waits on all of them before mounting
        # the GUI so the operator's first click lands on a warm pipeline.
        num_workers = self.config.envs.num_workers_per_gpu
        self.warmup_events: list = [mp.Event() for _ in range(num_workers)]

        # Single shutdown signal. Set by:
        #   - the debug GUI's "Terminate Program" button (external)
        #   - the supervisor when a slot exhausts its restart budget (internal)
        # The supervisor loop's wait() unblocks immediately when this fires,
        # so shutdown latency is microseconds rather than up to one heartbeat.
        self.shutdown_event = mp.Event()

        # Allocated in run().
        self.shm: Optional[SharedMemory] = None
        self.shm_config = None
        self._slots: list[_ProcessSlot] = []

    # ==================================================================
    # Public API
    # ==================================================================

    def run(self):
        """Main entry: allocate SHM, spawn processes, supervise until shutdown."""
        setup_logging(debug=self.config.envs.debug)
        try:
            self._init_resources()
            self._build_slots()
            self._start_all_slots()
            self._supervisor_loop()
        except KeyboardInterrupt:
            logger.info(f"[{self.name}] Interruption received (Ctrl+C).")
        except Exception as e:
            logger.critical(f"[{self.name}] Global dispatcher failure: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self.cleanup()

    def cleanup(self):
        """Strict resource reclamation. Idempotent — safe to call from any state."""
        logger.info(f"[{self.name}] Initiating graceful shutdown...")
        # Two passes: signal stop on every slot first, THEN join. This avoids
        # serializing the stop request behind each process's own shutdown
        # latency — they all wind down concurrently.
        for slot in self._slots:
            slot.stop()
        for slot in self._slots:
            slot.join_or_terminate()
        if self.shm is not None:
            self.shm.cleanup()

    # ==================================================================
    # Initialization
    # ==================================================================

    def _init_resources(self):
        logger.info(f"[{self.name}] Allocating Shared Memory...")
        # SharedMemory.build() = factory that constructs the nested config
        # from raw sizing params and returns an un-allocated instance.
        # allocate() actually creates the OS-level segments.
        shm = SharedMemory.build(
            name_prefix=self.config.shm.name_prefix,
            num_streams=self.config.num_streams,
            num_buffers=self.config.num_buffers,
            resolution=(self.config.shm.height, self.config.shm.width),
        )
        shm.allocate()
        self.shm = shm
        self.shm_config = shm.config

    def _build_slots(self):
        """One slot per child process. Factories close over current state."""
        logger.info(f"[{self.name}] Initializing system components...")
        num_workers = self.config.envs.num_workers_per_gpu

        # Declaration order is also start order. Consumer first so it's
        # monitoring the queue before any work arrives; grabber last so
        # data only flows once every downstream is ready.
        self._slots = [
            _ProcessSlot(key="consumer", label="ResultConsumer", factory=self._build_consumer),
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

    def _start_all_slots(self):
        for slot in self._slots:
            slot.start()

    # ------------------------------------------------------------------
    # Process factories — each returns a fresh, un-started Process
    # ------------------------------------------------------------------

    def _build_consumer(self) -> ResultProcess:
        # On restart, this rebuilds handlers too. Handler in-process state
        # (DisplayProcess, executors, audit fh) was tied to the dead consumer
        # PID, so spinning up fresh ones is the safe path. Reusing the
        # existing warmup events is fine: if workers are up they're already
        # set, so the new consumer's wait returns immediately.
        consumer = ResultProcess(
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
    # Supervisor
    # ==================================================================

    def _supervisor_loop(self):
        """
        Health-check + restart loop.

        Each iteration:
          1. Restart any dead slot, up to MAX_RESTARTS per slot.
          2. Sleep until the next heartbeat OR shutdown_event fires.
        """
        while not self.shutdown_event.is_set():
            for slot in self._slots:
                if slot.is_dead() and not self._try_restart(slot):
                    return
            # Interruptible sleep — shutdown_event.set() exits the loop now
            # instead of waiting out the heartbeat.
            self.shutdown_event.wait(timeout=_SUPERVISOR_INTERVAL)
        logger.info(f"[{self.name}] Shutdown requested; exiting supervisor loop.")

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
