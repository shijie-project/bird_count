import logging
import multiprocessing as mp

from runtime.config import Config
from runtime.shared_memory import SharedMemory, SharedMemoryConfig
from runtime.utils import setup_logging

from ._camera import CameraThread


_WATCHDOG_POLL_INTERVAL = 1.0  # how often the parent process checks thread health


logger = logging.getLogger(__name__)


class GrabberProcess(mp.Process):
    """
    Hosts one `CameraThread` per source, plus a watchdog that revives any
    thread that exits abnormally.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig):
        super().__init__(name="GrabberProcess")
        self.config = config
        self.shm_config = shm_config
        self.threads: list[CameraThread] = []
        self._stop_event = mp.Event()

    def run(self):
        setup_logging(self.config.envs.debug)
        logger.info(f"[{self.name}] Process started.")
        try:
            with SharedMemory(self.shm_config) as shm:
                self._spawn_threads(shm)
                self._watchdog_loop(shm)
                self._stop_threads()
        finally:
            logger.info(f"[{self.name}] Process stopped.")

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spawn_threads(self, shm: SharedMemory):
        for sid, source in enumerate(self.config.stream_sources):
            t = self._build_thread(sid, source, shm)
            self.threads.append(t)
            t.start()

    def _watchdog_loop(self, shm: SharedMemory):
        while not self._stop_event.is_set():
            for sid, t in enumerate(self.threads):
                if t.is_alive() or self._stop_event.is_set():
                    continue
                logger.warning(f"Thread {t.name} died. Reviving.")
                new_t = self._build_thread(sid, self.config.stream_sources[sid], shm)
                self.threads[sid] = new_t
                new_t.start()
            # Interruptible sleep — stop() exits the loop immediately
            # instead of waiting out the full poll interval.
            self._stop_event.wait(timeout=_WATCHDOG_POLL_INTERVAL)

    def _build_thread(self, sid: int, source: str, shm: SharedMemory) -> CameraThread:
        """Slice the per-stream SHM views and assemble a CameraThread."""
        return CameraThread(
            stream_id=sid,
            source=source,
            metadata=shm.stream_metadata[sid],
            frames=shm.stream_frames[sid],
            config=self.config,
        )

    def _stop_threads(self):
        # Two passes: signal everyone first, then join. Avoids serializing the
        # stop request behind each thread's interval.
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.join(timeout=1.0)
