import logging
import multiprocessing as mp
import queue

from runtime.shared_memory import SharedMemory

from ._render import _InternalMonitorRenderer


logger = logging.getLogger(__name__)


# Poll the queue at roughly the display rate so tick() fires even when no new
# batches arrive (otherwise pending frames would sit unrendered for up to the
# old 1.0 s timeout, and their SHM buffers would stay claimed).


class DisplayProcess(mp.Process):
    def __init__(self, config, shm_config, display_queue, ack_queue):
        super().__init__(name="Monitor", daemon=True)
        self.config = config
        self.shm_config = shm_config
        self.display_queue = display_queue
        self.ack_queue = ack_queue

    def _ack(self, pairs):
        """Send buffer indices back to ResultProcess so it can release SHM."""
        if not pairs or self.ack_queue is None:
            return
        try:
            self.ack_queue.put_nowait(pairs)
        except queue.Full:
            # Drop the ack — ResultProcess will force-release after timeout.
            logger.debug(f"[{self.name}] ack_queue full; relying on stale-ack sweep.")

    def run(self):
        shm_client = SharedMemory(self.shm_config, name=self.name)
        shm_client.connect()

        renderer = _InternalMonitorRenderer(num_streams=self.config.num_streams)
        logger.info(f"[{self.name}] Monitor Process started.")

        while True:
            # Stage phase: drain whatever's available. Coalesces multiple
            # batches per tick into one render pass per stream.
            try:
                batch = self.display_queue.get(timeout=renderer.ui_update_interval)
            except queue.Empty:
                batch = None
            except Exception as e:
                logger.error(f"[{self.name}] Queue error: {e}")
                batch = None

            if batch is not None:
                early_acks = []
                for result in batch.results:
                    ack = renderer.stage(result)
                    if ack is not None:
                        early_acks.append(ack)
                # Release buffers that were superseded or out-of-range right
                # away — no point holding them across the tick window.
                self._ack(early_acks)

            # Tick phase: render if the display interval has elapsed.
            try:
                render_acks = renderer.tick(shm_client)
            except Exception as e:
                logger.error(f"[{self.name}] Render error: {e}")
                render_acks = None
            self._ack(render_acks)
