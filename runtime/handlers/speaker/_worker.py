"""Sync façade the handler's thread pool submits to."""

import asyncio
import logging
from typing import Optional

from runtime.handlers._cancel import CancelSignal

from ._async import httpx, run_broadcast_lifecycle


logger = logging.getLogger(__name__)


# `audio_play?action=start` plays the clip once, so a sustained broadcast means
# re-issuing PLAY on a timer. This interval doubles as the worst-case latency
# between a cancel arriving and STOP going out (the wait is cancel-aware, so in
# practice only an in-flight request delays it).
_REPLAY_INTERVAL = 10.0
# Per-request network timeout. The speakers answer in well under a second when
# healthy; anything slower means a saturated link, not a slow device.
_REQUEST_TIMEOUT = 5.0
# Safety cap. A lifecycle that hits it stops the audio and returns the device to
# READY, so a lost cancel can't leave a speaker blaring overnight.
_MAX_BROADCAST_TIME = 3600.0
# Give up after this many consecutive PLAY failures — the speaker is almost
# certainly unreachable and retrying forever just pins a pool thread.
_MAX_CONSECUTIVE_PLAY_FAILURES = 5


# Re-exported so `_handler.py` imports everything it needs from `_worker`
# (`httpx` is None when the dependency is missing — the handler disables itself).
__all__ = ["BroadcastLifecycle", "CancelSignal", "httpx"]


class BroadcastLifecycle:
    """Sync façade around the async broadcast coroutine in `_async.py`.

    The handler builds one of these per activation and submits `.run()` to its
    thread pool. The handler owns the surrounding state machine and the
    cancel-signal registry; this object owns nothing but the parameters of a
    single broadcast.
    """

    def __init__(
        self,
        ip: str,
        auth: tuple[str, str],
        audio_file: str,
        cancel_signal: CancelSignal,
        tag: str = "Speaker",
        max_broadcast_time: Optional[float] = _MAX_BROADCAST_TIME,
        request_timeout: float = _REQUEST_TIMEOUT,
        replay_interval: float = _REPLAY_INTERVAL,
        max_consecutive_play_failures: int = _MAX_CONSECUTIVE_PLAY_FAILURES,
    ):
        self.ip = ip
        self.auth = auth
        self.audio_file = audio_file
        self.cancel_signal = cancel_signal
        self.tag = tag
        # Per-instance overrides — primarily for tests, which can't afford to
        # wait the full replay interval.
        self.max_broadcast_time = max_broadcast_time
        self.request_timeout = request_timeout
        self.replay_interval = replay_interval
        self.max_consecutive_play_failures = max_consecutive_play_failures

    def run(self) -> Optional[str]:
        """Synchronous entry point for the handler's ThreadPoolExecutor.

        Returns:
            None on clean completion, or a stringified exception message if the
            asyncio task crashed. The handler logs / audits the result.
        """
        try:
            asyncio.run(
                run_broadcast_lifecycle(
                    ip=self.ip,
                    auth=self.auth,
                    audio_file=self.audio_file,
                    cancel_signal=self.cancel_signal,
                    tag=self.tag,
                    max_broadcast_time=self.max_broadcast_time,
                    request_timeout=self.request_timeout,
                    replay_interval=self.replay_interval,
                    max_consecutive_play_failures=self.max_consecutive_play_failures,
                )
            )
            return None
        except Exception as e:
            logger.error("[%s] Lifecycle failed for %s: %s", self.tag, self.ip, e)
            return str(e)
