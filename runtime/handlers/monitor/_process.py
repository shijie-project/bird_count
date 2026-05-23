import logging
import queue
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Synchronized

from runtime.config import Config
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._render import _InternalMonitorRenderer


logger = logging.getLogger(__name__)


class DisplayProcess(Process):
    """Runs the OpenCV dashboard in its own process so cv2.waitKey doesn't stall
    the consumer thread. Polls `display_queue` at the renderer's update interval
    so `tick()` fires even between batches — otherwise pending SHM buffers would
    sit claimed for up to the queue's timeout."""

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        display_queue: Queue,
        ack_queue: Queue | None,
        show_density_map: Synchronized | None = None,
    ):
        super().__init__(name="Monitor", daemon=True)
        self.config = config
        self.shm_config = shm_config
        self.display_queue = display_queue
        self.ack_queue = ack_queue
        # Shared mp.Value(bool) — toggled by MonitorHandler from the consumer
        # thread, polled by the renderer each tick. None = always off.
        self.show_density_map = show_density_map

    def _ack(self, pairs) -> None:
        """Send buffer indices back to ResultProcess so it can release SHM."""
        if not pairs or self.ack_queue is None:
            return
        try:
            self.ack_queue.put_nowait(pairs)
        except queue.Full:
            # Drop the ack — ResultProcess will force-release after timeout.
            logger.debug("[%s] ack_queue full; relying on stale-ack sweep.", self.name)

    def run(self) -> None:
        shm_client = SharedMemory(self.shm_config, name=self.name)
        shm_client.connect()

        renderer = _InternalMonitorRenderer(
            num_streams=self.config.num_streams,
            show_density_map_flag=self.show_density_map,
        )
        logger.info("[%s] Monitor Process started.", self.name)

        while True:
            # Stage phase: drain whatever's available. Coalesces multiple
            # batches per tick into one render pass per stream.
            try:
                batch = self.display_queue.get(timeout=renderer.ui_update_interval)
            except queue.Empty:
                batch = None
            except Exception as e:
                logger.error("[%s] Queue error: %s", self.name, e, exc_info=True)
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
                logger.error("[%s] Render error: %s", self.name, e, exc_info=True)
                render_acks = None
            self._ack(render_acks)
