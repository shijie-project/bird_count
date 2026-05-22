"""VideoRecorderHandler — owns per-stream cv2.VideoWriter threads in-process.

One worker thread per enabled stream. `handle_batch` runs in the consumer
thread, copies each requested SHM frame, and pushes it onto the matching
per-stream `queue.Queue`. Each writer thread drains its queue into an mp4
segment via `_writer.writer_loop`.

Encoding is GIL-free (`cv2.VideoWriter.write` and `numpy.ndarray.copy` both
release the GIL), so N writer threads encode in parallel limited only by disk
I/O. No subprocess, no `mp.Queue`, no ack accounting — the handler is now
synchronous and never claims SHM buffers.
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from runtime.config import Config
from runtime.handlers import BaseHandler, GUIToggleMixin
from runtime.inferencer import BatchInferenceResult, InferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._writer import writer_loop


logger = logging.getLogger(__name__)


# Per-stream queue depth between handle_batch and the encoder thread.
# At 1080x720x3 uint8 each frame is ~2.3 MB → 30 frames ≈ 70 MB max per stream.
WRITER_QUEUE_MAXSIZE = 30

# Max time to wait for a writer thread to drain & finalize on disable/stop.
WRITER_JOIN_TIMEOUT = 5.0

# Log every Nth dropped frame so persistent disk-falling-behind is visible
# without flooding the log on a transient stall.
DROP_LOG_EVERY = 100


class VideoRecorderHandler(GUIToggleMixin, BaseHandler):
    """Owns one cv2.VideoWriter thread per enabled stream."""

    needs_frames = False  # we resolve SHM frames manually inside handle_batch

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        name: str = "VideoRecorder",
    ):
        super().__init__(config=config, shm_config=shm_config, name=name)

        self._all_stream_ids: set[int] = set(config.sid_to_ip.keys())
        self._enabled_streams: set[int] = set()
        self._initial_enabled: set[int] = set(self._all_stream_ids) if config.envs.enable_video_recorder else set()

        # Encoder parameters — snapshot once; config doesn't change at runtime.
        self._fps = float(config.fps)
        self._frame_size = (config.shm.width, config.shm.height)  # (W, H) for cv2
        self._segment_seconds = float(config.envs.video_segment_seconds)
        self._output_dir = Path(config.envs.video_record_dir)

        # Per-stream thread state — all access is from the consumer thread.
        self._worker_queues: dict[int, queue.Queue] = {}
        self._worker_stops: dict[int, threading.Event] = {}
        self._worker_threads: dict[int, threading.Thread] = {}
        self._drop_counts: dict[int, int] = {}

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
        if self._started:
            self._spawn_writer(stream_id)
        self.audit.log("recorder.stream_enable", stream_id=stream_id)
        logger.info(f"[{self.name}] Recording ON for stream {stream_id}.")
        return True

    def disable_stream(self, stream_id: int) -> bool:
        if stream_id not in self._enabled_streams:
            return False
        self._enabled_streams.discard(stream_id)
        self._shutdown_writer(stream_id)
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
        self._started = True
        for sid in sorted(self._initial_enabled):
            self.enable_stream(sid)

    def stop(self) -> None:
        for sid in list(self._enabled_streams):
            self._shutdown_writer(sid)
        self._enabled_streams.clear()
        self._started = False
        super().stop()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        # Unused — handle_batch is overridden.
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemory) -> set[tuple[int, int]]:
        """Copy enabled-stream frames out of SHM into the writer queues.

        Synchronous — returns an empty claim set so the consumer can mark
        each buffer FREE immediately. The writer threads operate on the
        local `.copy()` taken below, so subsequent grabber overwrites
        cannot corrupt in-flight encodes.
        """
        if not self._enabled_streams or not batch_result.results:
            return set()

        frames = shm_client.frames
        now = time.time()
        for r in batch_result.results:
            sid = int(r.stream_id)
            if sid not in self._enabled_streams:
                continue
            wq = self._worker_queues.get(sid)
            if wq is None:
                continue  # stream disabled between dispatch and here
            frame = frames[sid, int(r.buffer_idx)].copy()
            try:
                wq.put_nowait((frame, now))
            except queue.Full:
                self._drop_counts[sid] = self._drop_counts.get(sid, 0) + 1
                n = self._drop_counts[sid]
                if n == 1 or n % DROP_LOG_EVERY == 0:
                    logger.warning(
                        f"[{self.name}] Stream {sid} queue full (drops: {n}). Disk encoder is falling behind."
                    )
        return set()

    # ------------------------------------------------------------------
    # Writer thread plumbing
    # ------------------------------------------------------------------

    def _spawn_writer(self, sid: int) -> None:
        if sid in self._worker_threads:
            return
        q: queue.Queue = queue.Queue(maxsize=WRITER_QUEUE_MAXSIZE)
        ev = threading.Event()
        t = threading.Thread(
            target=writer_loop,
            args=(
                sid,
                q,
                ev,
                self._fps,
                self._frame_size,
                self._segment_seconds,
                self._output_dir,
                self.audit,
            ),
            name=f"VideoWriter-{sid:02d}",
            daemon=True,
        )
        self._worker_queues[sid] = q
        self._worker_stops[sid] = ev
        self._worker_threads[sid] = t
        self._drop_counts[sid] = 0
        t.start()
        logger.info(f"[{self.name}] Stream {sid} writer thread started.")

    def _shutdown_writer(self, sid: int) -> None:
        ev = self._worker_stops.pop(sid, None)
        q = self._worker_queues.pop(sid, None)
        t = self._worker_threads.pop(sid, None)
        drops = self._drop_counts.pop(sid, 0)

        if ev is not None:
            ev.set()
        if q is not None:
            try:
                q.put_nowait(None)  # writer_loop's sentinel for post-stop drain
            except queue.Full:
                pass
        if t is not None:
            t.join(timeout=WRITER_JOIN_TIMEOUT)
            if t.is_alive():
                logger.warning(
                    f"[{self.name}] Writer thread for stream {sid} did not join in "
                    f"{WRITER_JOIN_TIMEOUT}s; segment may be incomplete."
                )
        if drops:
            logger.warning(f"[{self.name}] Stream {sid} dropped {drops} frame(s).")
        logger.info(f"[{self.name}] Stream {sid} writer thread stopped.")
