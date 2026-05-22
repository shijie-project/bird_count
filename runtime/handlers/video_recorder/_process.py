import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from runtime.audit import AuditLog
from runtime.config import Config
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._writer import writer_loop


logger = logging.getLogger(__name__)


# Per-stream in-process queue depth (inside VideoWriterProcess). Buffers
# transient disk stalls without ballooning RAM. At 1080x720x3 uint8 that's
# ~2.3 MB/frame → 30 frames ≈ 70 MB max per active stream.
WRITER_QUEUE_MAXSIZE = 30

# Max time to wait for a per-stream writer thread to drain & finalize.
WRITER_JOIN_TIMEOUT = 5.0

# Log every Nth dropped frame so persistent disk-falling-behind is visible
# without flooding the log on a transient stall.
DROP_LOG_EVERY = 100


class VideoWriterProcess(mp.Process):
    """
    Dedicated process that owns all per-stream cv2.VideoWriters.

    Protocol (work_queue): small typed tuples
        ('enable',  sid)                  -- spawn worker thread for stream
        ('disable', sid)                  -- drain + finalize that stream's segment
        ('frames',  [(sid, b_idx), ...])  -- batch of buffer pairs to record
        ('stop',)                         -- clean shutdown sentinel

    On 'frames' this process:
      1. Reads each frame from SHM (the handler claimed those buffers, so they
         are guaranteed to be in READING state).
      2. `.copy()` into an in-process per-stream queue.
      3. Acks the pairs to `ack_queue` so ResultProcess can release SHM.

    Disk I/O happens on per-stream worker threads — N streams = N parallel
    encoders, limited by disk write throughput rather than ResultProcess.
    """

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        work_queue: mp.Queue,
        ack_queue: Optional[mp.Queue],
    ):
        super().__init__(name="VideoWriter", daemon=True)
        self.fps = float(config.fps)
        self.frame_size = (config.shm.width, config.shm.height)  # (W, H)
        self.segment_seconds = float(getattr(config.envs, "video_segment_seconds", 300.0))
        self.output_dir = Path(getattr(config.envs, "video_record_dir", "recordings"))
        self.shm_config = shm_config
        self.work_queue = work_queue
        self.ack_queue = ack_queue
        self._audit_log_path = config.envs.audit_log_path

    def _ack(self, pairs):
        if not pairs or self.ack_queue is None:
            return
        try:
            self.ack_queue.put_nowait(pairs)
        except queue.Full:
            logger.debug(f"[{self.name}] ack_queue full; relying on stale-ack sweep.")

    def run(self):
        shm_client = SharedMemory(self.shm_config)
        shm_client.connect()

        audit = AuditLog(self._audit_log_path, name=self.name)
        audit.log("process.start", name=self.name)
        logger.info(f"[{self.name}] Writer Process started.")

        worker_queues: dict[int, queue.Queue] = {}
        worker_threads: dict[int, threading.Thread] = {}
        worker_stops: dict[int, threading.Event] = {}
        drop_counts: dict[int, int] = {}

        def _spawn(sid: int):
            if sid in worker_threads:
                return
            q: queue.Queue = queue.Queue(maxsize=WRITER_QUEUE_MAXSIZE)
            ev = threading.Event()
            t = threading.Thread(
                target=writer_loop,
                args=(
                    sid,
                    q,
                    ev,
                    self.fps,
                    self.frame_size,
                    self.segment_seconds,
                    self.output_dir,
                    audit,
                ),
                name=f"VideoWriter-{sid:02d}",
                daemon=True,
            )
            worker_queues[sid] = q
            worker_stops[sid] = ev
            worker_threads[sid] = t
            drop_counts[sid] = 0
            t.start()
            audit.log("recorder.stream_enable", stream_id=sid)
            logger.info(f"[{self.name}] Stream {sid} writer thread started.")

        def _shutdown(sid: int):
            ev = worker_stops.pop(sid, None)
            q = worker_queues.pop(sid, None)
            t = worker_threads.pop(sid, None)
            drops = drop_counts.pop(sid, 0)
            if ev is not None:
                ev.set()
            if q is not None:
                try:
                    q.put_nowait(None)
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
            audit.log("recorder.stream_disable", stream_id=sid, dropped=drops)
            logger.info(f"[{self.name}] Stream {sid} writer thread stopped.")

        try:
            while True:
                try:
                    msg = self.work_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"[{self.name}] work_queue error: {e}")
                    continue

                kind = msg[0]
                if kind == "stop":
                    break
                elif kind == "enable":
                    _spawn(int(msg[1]))
                elif kind == "disable":
                    _shutdown(int(msg[1]))
                elif kind == "frames":
                    pairs = msg[1]
                    for sid, b_idx in pairs:
                        sid = int(sid)
                        b_idx = int(b_idx)
                        q = worker_queues.get(sid)
                        if q is None:
                            continue  # stream disabled between enqueue and dispatch
                        # SHM still in READING state thanks to the handler's claim,
                        # but the worker thread runs later — copy before acking.
                        frame = shm_client.frames[sid, b_idx].copy()
                        try:
                            q.put_nowait((frame, time.time()))
                        except queue.Full:
                            drop_counts[sid] = drop_counts.get(sid, 0) + 1
                            n = drop_counts[sid]
                            if n == 1 or n % DROP_LOG_EVERY == 0:
                                logger.warning(
                                    f"[{self.name}] Stream {sid} queue full "
                                    f"(drops: {n}). Disk encoder is falling behind."
                                )
                    # Always ack — frames are copied or dropped, but the SHM slot
                    # is no longer needed regardless.
                    self._ack(pairs)
                else:
                    logger.warning(f"[{self.name}] Unknown work-queue message: {msg!r}")
        finally:
            # Clean shutdown: finalize every active segment before exiting.
            for sid in list(worker_threads):
                _shutdown(sid)
            shm_client.disconnect()
            audit.log("process.stop", name=self.name)
            audit.close()
            logger.info(f"[{self.name}] Writer Process stopped.")
