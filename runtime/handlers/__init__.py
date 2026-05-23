"""Handler package.

Re-exports the base classes and the `init_handlers` factory. Optional handlers
(smart plug, speaker) are imported lazily inside the factory so a missing
dependency does not stall package import.
"""

import logging
from multiprocessing import Queue

from runtime.config import Config
from runtime.shared_memory import SharedMemoryConfig

from ._base import BaseHandler, GUIToggleMixin
from .monitor import MonitorHandler
from .video_recorder import VideoRecorderHandler


logger = logging.getLogger(__name__)


__all__ = [
    "BaseHandler",
    "GUIToggleMixin",
    "MonitorHandler",
    "VideoRecorderHandler",
    "init_handlers",
]


def init_handlers(
    config: Config,
    shm_config: SharedMemoryConfig,
    ack_queue: Queue | None = None,
) -> list[BaseHandler]:
    """Instantiate every enabled handler in dispatch order.

    Monitor + video recorder are always registered (runtime-toggleable via GUI).
    Optional handlers (smart plug, speaker) are loaded only when their env flag
    is on — lazy import keeps startup independent of their dependencies.
    """
    handlers: list[BaseHandler] = []

    def _register(handler: BaseHandler, label: str) -> None:
        handlers.append(handler)
        logger.info("Handler Registered: %s", label)

    _register(MonitorHandler(config, shm_config, ack_queue=ack_queue), "Monitor")
    _register(VideoRecorderHandler(config, shm_config), "VideoRecorder")

    if config.envs.enable_smart_plug:
        from .smart_plug import SmartPlugHandler

        _register(SmartPlugHandler(config, shm_config), "Smart Plug")

    if config.envs.enable_speaker:
        from .speaker import SpeakerHandler

        _register(SpeakerHandler(config), "Speaker")

    return handlers
