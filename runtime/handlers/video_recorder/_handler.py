"""VideoRecorderHandler — owns the VideoWriterProcess and the per-stream enable set."""

import logging
import multiprocessing as mp
import queue
from typing import Optional

import numpy as np

from runtime.config import Config
from runtime.handlers import BaseHandler, GUIToggleMixin
from runtime.inference_process import BatchInferenceResult, InferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._process import VideoWriterProcess


logger = logging.getLogger(__name__)


# Cross-process work queue depth. Items are small typed tuples (a few KB), so
# we can be generous — this only buffers a producer-faster-than-consumer spike.
WORK_QUEUE_MAXSIZE = 256

# Max time to wait for the writer process to exit cleanly after `stop`.
PROC_JOIN_TIMEOUT = 15.0
PROC_TERMINATE_GRACE = 1.0
PROC_KILL_GRACE = 0.5

# Timeout for sending an enable/disable control message into the work queue.
CONTROL_SEND_TIMEOUT = 1.0


class VideoRecorderHandler(GUIToggleMixin, BaseHandler):
    """Proxy handler that forwards work to a VideoWriterProcess.

    Holds the canonical per-stream enable set and a `mp.Queue` to the writer
    process. All disk I/O, segment rotation, and per-stream parallelism live
    inside VideoWriterProcess — this handler is just a router with a small
    GUI surface. Mirrors the MonitorHandler / DisplayProcess split.

    The writer process is always-on while `start()` is in effect. With no
    streams enabled it sits idle (sub-percent CPU). Aggregate enable/disable
    fans out to every known stream; per-stream toggles can be flipped
    independently from the debug GUI.
    """

    needs_frames = False  # The writer process resolves SHM frames itself.

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        ack_queue: Optional[mp.Queue] = None,
        name: str = "VideoRecorder",
    ):
        super().__init__(config=config, shm_config=shm_config, name=name)
        self.ack_queue = ack_queue

        self._all_stream_ids: set[int] = set(config.sid_to_ip.keys())
        self._enabled_streams: set[int] = set()
        # Streams to enable when start() fires. Initially `enable_video_recorder`
        # is a global on/off — present-but-empty means "off"; full set means "on".
        self._initial_enabled: set[int] = (
            set(self._all_stream_ids) if bool(getattr(config.envs, "enable_video_recorder", False)) else set()
        )

        self.work_queue: Optional[mp.Queue] = None
        self.proc: Optional[VideoWriterProcess] = None
        self._started = False

    # ------------------------------------------------------------------
    # GUI surface  (matches runtime.gui._base.RecorderHandler)
    # ------------------------------------------------------------------

    def enable(self) -> bool:
        for sid in self._all_stream_ids:
            self.enable_stream(sid)
        return True

    def disable(self) -> bool:
        if not self._enabled_streams:
            return False
        for sid in list(self._enabled_streams):
            self.disable_stream(sid)
        return False

    def is_enabled(self) -> bool:
        return bool(self._enabled_streams)

    def enable_stream(self, stream_id: int) -> bool:
        if stream_id in self._enabled_streams:
            return True
        self._enabled_streams.add(stream_id)
        self._send(("enable", stream_id))
        self.audit.log("recorder.stream_enable", stream_id=stream_id)
        logger.info(f"[{self.name}] Recording ON for stream {stream_id}.")
        return True

    def disable_stream(self, stream_id: int) -> bool:
        if stream_id not in self._enabled_streams:
            return False
        self._enabled_streams.discard(stream_id)
        self._send(("disable", stream_id))
        self.audit.log("recorder.stream_disable", stream_id=stream_id)
        logger.info(f"[{self.name}] Recording OFF for stream {stream_id}.")
        return False

    def is_stream_enabled(self, stream_id: int) -> bool:
        return stream_id in self._enabled_streams

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()
        self.audit.log(
            "handler.start",
            handler=self.name,
            initial_streams=sorted(self._initial_enabled),
        )
        self._spawn_writer()
        self._started = True
        for sid in sorted(self._initial_enabled):
            self.enable_stream(sid)

    def stop(self) -> None:
        self._terminate_writer()
        super().stop()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        # Unused — handle_batch is overridden. Kept satisfied for clarity only.
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemory) -> set[tuple[int, int]]:
        if not self._enabled_streams or self.work_queue is None or not batch_result.results:
            return set()
        active = [
            (int(r.stream_id), int(r.buffer_idx)) for r in batch_result.results if r.stream_id in self._enabled_streams
        ]
        if not active:
            return set()
        try:
            self.work_queue.put_nowait(("frames", active))
        except queue.Full:
            # Don't claim — buffers are released and the frame is lost.
            logger.warning(f"[{self.name}] work_queue full; dropping {len(active)} frame(s).")
            return set()
        return set(active)

    # ------------------------------------------------------------------
    # Writer process plumbing
    # ------------------------------------------------------------------

    def _spawn_writer(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        self.work_queue = mp.Queue(maxsize=WORK_QUEUE_MAXSIZE)
        self.proc = VideoWriterProcess(self.config, self.shm_config, self.work_queue, self.ack_queue)
        self.proc.start()
        logger.info(f"[{self.name}] Writer process spawned.")

    def _terminate_writer(self) -> None:
        proc, wq = self.proc, self.work_queue
        self.proc = None
        self.work_queue = None
        if proc is None:
            return
        # Polite shutdown: sentinel + join. The writer drains queues and
        # finalizes each segment in its finally-clause.
        if wq is not None:
            try:
                wq.put(("stop",), timeout=1.0)
            except Exception:
                pass
        try:
            proc.join(timeout=PROC_JOIN_TIMEOUT)
            if proc.is_alive():
                logger.warning(f"[{self.name}] Writer did not exit in {PROC_JOIN_TIMEOUT}s; terminating.")
                proc.terminate()
                proc.join(timeout=PROC_TERMINATE_GRACE)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=PROC_KILL_GRACE)
        except Exception as e:
            logger.error(f"[{self.name}] Error joining writer process: {e}")
        logger.info(f"[{self.name}] Writer process stopped.")

    def _send(self, msg: tuple) -> None:
        """Push a control message to the writer; tolerate brief queue saturation."""
        if self.work_queue is None:
            return
        try:
            self.work_queue.put(msg, timeout=CONTROL_SEND_TIMEOUT)
        except queue.Full:
            logger.warning(f"[{self.name}] work_queue full while sending {msg[0]!r}.")
