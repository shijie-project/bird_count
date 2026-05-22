"""ResultConsumer — in-process consumer for the inference workers' batched output.

Used to be `ResultProcess(mp.Process)`; merged into the dispatcher's main thread.
The consumer exposes `setup` / `tick` / `cleanup`, and the dispatcher's main
loop calls `tick()` every iteration. GUI ownership stays with
`ResultGUIController` (see `_gui_controller.py`).

Per-tick recipe (one call to `tick()`):
  1. Pump the GUI.
  2. Drain async-handler acks; force-release stale claims.
  3. Pull one BatchInferenceResult off `result_queue` (10 ms timeout).
  4. Evaluate alerts in place.
  5. Dispatch to every handler; collect SHM claims for async ack.
  6. Free unclaimed slots immediately; defer claimed ones.
"""

import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Iterable
from typing import Optional, TypeVar

from runtime.audit import AuditLog
from runtime.config import Config
from runtime.handlers import BaseHandler, MonitorHandler, VideoRecorderHandler
from runtime.inferencer import BatchInferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._ack_registry import _PendingAckRegistry
from ._alert_evaluator import _AlertEvaluator
from ._gui_controller import ResultGUIController


logger = logging.getLogger(__name__)


# Tick budgets (seconds).
_QUEUE_POLL_TIMEOUT = 0.01  # result_queue.get timeout → inner-loop tick
_STALE_SWEEP_INTERVAL = 1.0  # minimum gap between stale-ack sweeps

# Sentinel buffer_idx emitted by an inference worker when no streams were READY
# this tick. Such batches carry no real SHM claims, so the ack accounting and
# release paths short-circuit on this value.
_SENTINEL_BUFFER_IDX = -1


H = TypeVar("H", bound=BaseHandler)


class ResultConsumer:
    """In-process consumer for inference batches.

    Owns: handlers, alert evaluator, ack registry, audit log, GUI controller.
    Does NOT own: SHM client (passed in by the dispatcher), the result queue
    pump (the dispatcher's main loop calls `tick()`), or process supervision.
    """

    ACK_TIMEOUT_SEC = 5.0
    # Upper bound for the warmup wait. Long enough for cudnn autotune across
    # every batch size, short enough that the GUI eventually appears even if
    # a worker is silently stuck.
    WARMUP_WAIT_TIMEOUT_SEC = 300.0

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        result_queue: mp.Queue,
        ack_queue: Optional[mp.Queue] = None,
        warmup_events: "Iterable[mp.synchronize.Event]" = (),
        shutdown_event: "Optional[mp.synchronize.Event]" = None,
        name: str = "ResultConsumer",
    ):
        self.name = name
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self.warmup_events = tuple(warmup_events)
        self.shutdown_event = shutdown_event  # set by debug GUI's Terminate button

        self.handlers: list[BaseHandler] = []

        # Pure-logic sub-components (no SHM / process coupling).
        self.alerts = _AlertEvaluator(
            stream_thresholds=config.sid_to_threshold,
            trigger_delay=float(config.envs.alert_trigger_delay),
        )
        self.acks = _PendingAckRegistry(
            ack_queue=ack_queue,
            ack_timeout_sec=self.ACK_TIMEOUT_SEC,
            name=self.name,
        )

        # Audit log is opened in `setup()`. Pre-bind a disabled instance so
        # callbacks can call self.audit.log unconditionally beforehand.
        self.audit: AuditLog = AuditLog(None, name=self.name)

        # Built in setup() once handlers have started and SHM is wired up.
        self.gui: Optional[ResultGUIController] = None

        # Stale-sweep clock — initialised in setup() so the first tick after
        # mount doesn't immediately sweep a registry that hasn't received
        # any claims yet.
        self._last_stale_sweep = 0.0

    # ==================================================================
    # Public API
    # ==================================================================

    def register_handler(self, handler: BaseHandler) -> None:
        """Register a handler. Order is preserved across dispatch."""
        self.handlers.append(handler)

    def setup(self, shm: SharedMemory) -> None:
        """Open audit log, start handlers, wait for warmup, mount GUI."""
        self.audit = AuditLog(self.config.envs.audit_log_path, name=self.name)
        self.audit.log("consumer.start", name=self.name)

        for h in self.handlers:
            h.start()

        self._wait_for_inference_warmup()
        self.gui = self._build_gui()
        self.gui.setup()

        self._last_stale_sweep = time.time()

    def tick(self, shm: SharedMemory) -> None:
        """One iteration of the consumption loop. Called by the dispatcher's main loop."""
        if self.gui is not None:
            self.gui.pump()

        self.acks.drain(shm)

        now = time.time()
        if now - self._last_stale_sweep >= _STALE_SWEEP_INTERVAL:
            self.acks.sweep_stale(shm, now)
            self._last_stale_sweep = now

        try:
            batch_packet: BatchInferenceResult = self.result_queue.get(timeout=_QUEUE_POLL_TIMEOUT)
        except queue.Empty:
            return

        self._process_batch(shm, batch_packet)

    def cleanup(self, shm: SharedMemory) -> None:
        """Tear down GUI, stop handlers, drain final acks, close audit."""
        if self.gui is not None:
            self.gui.destroy()
            self.gui = None

        for h in self.handlers:
            try:
                h.stop()
            except Exception as e:
                logger.error(f"[{self.name}] {type(h).__name__}.stop() failed: {e}")

        # Release any final claims so the SHM ring is clean for the next run.
        self.acks.drain(shm)
        self.acks.flush(shm)

        self.audit.log("consumer.stop", name=self.name)
        self.audit.close()
        logger.info(f"[{self.name}] Stopped.")

    # ==================================================================
    # GUI callbacks (invoked by ResultGUIController on this thread)
    # ==================================================================

    def _on_gui_cancel_all(self) -> None:
        """Reset alert state and ask every handler to abort in-flight lifecycles."""
        self.alerts.reset_holddown()
        for h in self.handlers:
            try:
                h.cancel_all()
            except Exception as e:
                logger.error(f"[{self.name}] cancel_all failed on {type(h).__name__}: {e}")

    def _on_gui_terminate(self) -> None:
        """Signal the dispatcher to shut down (or warn if no signal was wired)."""
        if self.shutdown_event is None:
            logger.warning(f"[{self.name}] Terminate requested but no shutdown_event was wired.")
            return
        self.shutdown_event.set()

    # ==================================================================
    # Utilities
    # ==================================================================

    def _find_handler(self, handler_type: type[H]) -> Optional[H]:
        """First registered handler matching `handler_type`, or None."""
        return next((h for h in self.handlers if isinstance(h, handler_type)), None)

    def _get_active_devices_snapshot(self) -> dict[str, set[str]]:
        """Aggregate active-device sets across handlers for GUI status display."""
        snapshot: dict[str, set[str]] = {}
        for h in self.handlers:
            try:
                snapshot[type(h).__name__] = h.get_active_devices()
            except Exception:
                snapshot[type(h).__name__] = set()
        return snapshot

    def _build_gui(self) -> ResultGUIController:
        """Wire ResultGUIController to this process's handlers + audit log."""
        monitor_handler = self._find_handler(MonitorHandler)
        return ResultGUIController(
            config=self.config,
            audit=self.audit,
            debug_mode=bool(self.config.envs.debug),
            on_cancel_all=self._on_gui_cancel_all,
            on_terminate=self._on_gui_terminate,
            monitor_handler=monitor_handler,
            recorder_handler=self._find_handler(VideoRecorderHandler),
            density_map_handler=monitor_handler,
            active_devices_provider=self._get_active_devices_snapshot,
        )

    def _wait_for_inference_warmup(self) -> None:
        """Block until every inference worker signals warmup-complete (with timeout)."""
        if not self.warmup_events:
            return
        n = len(self.warmup_events)
        logger.info(f"[{self.name}] Waiting for {n} inference worker(s) to finish warmup before mounting GUI...")

        deadline = time.time() + self.WARMUP_WAIT_TIMEOUT_SEC
        for i, ev in enumerate(self.warmup_events):
            while True:
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    return
                remaining = deadline - time.time()
                if remaining <= 0:
                    logger.warning(
                        f"[{self.name}] Timed out after {self.WARMUP_WAIT_TIMEOUT_SEC:.0f}s waiting "
                        f"for InferenceWorker-{i}; mounting GUI anyway."
                    )
                    return
                # Short polling slice so a shutdown signal is noticed promptly.
                if ev.wait(timeout=min(0.5, remaining)):
                    break

        logger.info(f"[{self.name}] Inference workers ready. Mounting GUI.")

    # ==================================================================
    # Per-batch processing
    # ==================================================================

    def _process_batch(self, shm: SharedMemory, batch_packet: BatchInferenceResult) -> None:
        """Evaluate alerts → dispatch to handlers → release/claim SHM slots."""
        claimed: set[tuple[int, int]] = set()
        try:
            overrides = self.gui.manual_override_streams if self.gui is not None else set()
            self.alerts.evaluate_inplace(batch_packet, overrides, self.audit)
            claimed = self._dispatch_handlers(shm, batch_packet)
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected processing error: {e}")
        finally:
            self._release_or_claim(shm, batch_packet, claimed)

    def _dispatch_handlers(self, shm: SharedMemory, batch_packet: BatchInferenceResult) -> set[tuple[int, int]]:
        """Run every handler; union the (sid, b_idx) pairs they claim for async ack."""
        claimed: set[tuple[int, int]] = set()
        for handler in self.handlers:
            try:
                pairs = handler.handle_batch(batch_packet, shm)
                if pairs:
                    claimed |= pairs
            except Exception as e:
                logger.error(f"[{self.name}] Handler {type(handler).__name__} error: {e}")
        return claimed

    def _release_or_claim(
        self,
        shm: SharedMemory,
        batch_packet: BatchInferenceResult,
        claimed: set[tuple[int, int]],
    ) -> None:
        """Free unclaimed slots immediately; defer claimed ones until ack."""
        b_idxs = batch_packet.buffer_indices
        # Sentinel batches carry no real buffers — nothing to release or claim.
        if not b_idxs or b_idxs[0] == _SENTINEL_BUFFER_IDX:
            return

        now = time.time()
        claimed_pairs: list[tuple[int, int]] = []
        to_release: list[tuple[int, int]] = []
        for s, b in zip(batch_packet.stream_ids, b_idxs):
            key = (int(s), int(b))
            if key in claimed:
                claimed_pairs.append(key)
            else:
                to_release.append(key)

        self.acks.claim(claimed_pairs, now)
        _PendingAckRegistry.release_buffers(shm, to_release)
