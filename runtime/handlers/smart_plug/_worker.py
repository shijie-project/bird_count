import asyncio
import logging
from typing import Optional

from ._async import ApiClient, CancelSignal, run_plug_lifecycle


logger = logging.getLogger(__name__)


# Per-step network timeouts. ON/OFF can be slow when a plug just woke from
# sleep; info polling is cheap and should fail fast so we keep cycling.
_ON_OFF_TIMEOUT = 10.0
_POLL_TIMEOUT = 5.0
# How often the lifecycle wakes up to re-check the plug's hardware state.
# Doubles as the maximum latency between a cancel signal arriving and the
# OFF command actually being issued.
_POLL_INTERVAL = 5.0
# Bail out of the lifecycle after this many consecutive poll failures; the
# plug is almost certainly unreachable and continuing to spin wastes a thread.
_MAX_CONSECUTIVE_POLL_FAILURES = 5


# Re-exported so `_handler.py` can keep importing both names from `_worker`.
__all__ = ["ApiClient", "CancelSignal", "PlugLifecycle"]


class PlugLifecycle:
    """Sync façade around the async plug-driving coroutine in `_async.py`.

    The handler builds one of these per activation and submits `.run()` to
    its thread pool. The handler owns the surrounding state machine and the
    cancel-signal registry; the lifecycle creates its own `CancelSignal`
    inside the asyncio loop and hands it back via `register_signal`.
    """

    def __init__(
        self,
        ip: str,
        api_client: ApiClient,
        cancel_signal: CancelSignal,
        max_idle_time: Optional[float] = None,
        tag: str = "SmartPlug",
        on_off_timeout: float = _ON_OFF_TIMEOUT,
        poll_timeout: float = _POLL_TIMEOUT,
        poll_interval: float = _POLL_INTERVAL,
        max_consecutive_poll_failures: int = _MAX_CONSECUTIVE_POLL_FAILURES,
    ):
        self.ip = ip
        self.api_client = api_client
        self.cancel_signal = cancel_signal
        self.max_idle_time = max_idle_time
        self.tag = tag
        # Per-instance overrides — primarily for tests, which can't afford to
        # wait the full 5s poll interval.
        self.on_off_timeout = on_off_timeout
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.max_consecutive_poll_failures = max_consecutive_poll_failures

    def run(self) -> Optional[str]:
        """Synchronous entry point for the handler's ThreadPoolExecutor.

        Returns:
            None on clean completion, or a stringified exception message if
            the asyncio task crashed. The handler logs/audits the result.
        """
        try:
            asyncio.run(
                run_plug_lifecycle(
                    ip=self.ip,
                    api_client=self.api_client,
                    cancel_signal=self.cancel_signal,
                    max_idle_time=self.max_idle_time,
                    tag=self.tag,
                    on_off_timeout=self.on_off_timeout,
                    poll_timeout=self.poll_timeout,
                    poll_interval=self.poll_interval,
                    max_consecutive_poll_failures=self.max_consecutive_poll_failures,
                )
            )
            return None
        except Exception as e:
            logger.error(f"[{self.tag}] Lifecycle failed for {self.ip}: {e}")
            return str(e)
