"""Async transport + broadcast lifecycle for the network speakers.

`SpeakerClient` is a thin wrapper over the device's `/cgi-bin` audio API
(`audio_play` / `audio_get_status`), and `run_broadcast_lifecycle` is the
one-shot coroutine that keeps a single speaker broadcasting until the operator
cancels, stops it at the panel, or the safety timeout expires.

Split out of `_handler.py` for the same reason as the smart plug: the handler
owns the device state machine and the thread pool, this module owns the network
protocol, the poll cadence, and the retry rules.
"""

import logging
import math
import time
from typing import Optional

from runtime.handlers._cancel import CancelSignal


try:
    import httpx
except ImportError:  # pragma: no cover - depends on deployment
    httpx = None


logger = logging.getLogger(__name__)


# STOP is safety-critical — retry once on transient failure before giving up.
# (The timing tunables live in `_worker.py`, which passes them in.)
_STOP_RETRY_ATTEMPTS = 2

# Substrings that mark the device as *not currently playing* in the plaintext
# body returned by `audio_get_status`.
_IDLE_TOKENS = ("stopped", "idle")


class SpeakerClient:
    """Async wrapper over one speaker's `/cgi-bin` audio API.

    Borrows the caller's `httpx.AsyncClient` (which carries the basic-auth
    credentials and the timeout) instead of owning one, so a lifecycle opens
    exactly one connection pool for its whole run.
    """

    def __init__(self, ip: str, client: "httpx.AsyncClient", audio_file: str, tag: str = "Speaker"):
        self.ip = ip
        self.tag = tag
        self.audio_file = audio_file
        self._client = client
        self._base_url = f"http://{ip}/cgi-bin"

    async def play(self) -> None:
        """Start the configured clip. Raises on any transport / non-2xx error."""
        await self._get("audio_play", {"name": self.audio_file, "action": "start", "time": 1})

    async def stop(self) -> None:
        """Stop the configured clip. Raises on any transport / non-2xx error."""
        await self._get("audio_play", {"name": self.audio_file, "action": "stop", "time": 1})

    async def is_idle(self) -> Optional[bool]:
        """Whether the device reports itself as not playing.

        Returns None when the status endpoint is unreachable or answers with
        something we don't recognise — the caller treats that as "unknown" and
        keeps broadcasting, because dropping a deterrent on a parse failure is
        the worse error.
        """
        try:
            response = await self._get("audio_get_status")
        except Exception as e:
            logger.debug("[%s] Status poll failed on %s: %s", self.tag, self.ip, e)
            return None
        body = response.text.lower()
        if any(token in body for token in _IDLE_TOKENS):
            return True
        return False

    async def _get(self, endpoint: str, params: Optional[dict] = None) -> "httpx.Response":
        response = await self._client.get(f"{self._base_url}/{endpoint}", params=params)
        response.raise_for_status()
        return response


async def _stop_audio(speaker: SpeakerClient, *, reason: str) -> None:
    """Send STOP with retry. Raises on final failure so the caller is notified
    that the speaker may still be broadcasting."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, _STOP_RETRY_ATTEMPTS + 1):
        try:
            await speaker.stop()
            logger.info("[%s] %s STOP issued (%s).", speaker.tag, speaker.ip, reason)
            return
        except Exception as e:
            last_exc = e
            if attempt < _STOP_RETRY_ATTEMPTS:
                logger.warning(
                    "[%s] STOP attempt %d failed on %s (%s), retrying: %s", speaker.tag, attempt, speaker.ip, reason, e
                )
            else:
                logger.error(
                    "[%s] STOP failed after %d attempts on %s (%s): %s", speaker.tag, attempt, speaker.ip, reason, e
                )
    assert last_exc is not None
    raise last_exc


async def run_broadcast_lifecycle(
    *,
    ip: str,
    auth: tuple[str, str],
    audio_file: str,
    cancel_signal: CancelSignal,
    tag: str,
    max_broadcast_time: Optional[float],
    request_timeout: float,
    replay_interval: float,
    max_consecutive_play_failures: int,
) -> None:
    """One-shot async lifecycle for a single speaker.

    The caller passes a pre-created `cancel_signal`; we bind it to the running
    loop on entry and unbind on exit, so any `request()` that arrived before
    bind() is carried forward automatically by the signal.

    Flow:
      1. Issue PLAY.
      2. Loop until one of:
           - `cancel_signal` is set (GUI cancel_all) → issue explicit STOP.
           - the device reports idle mid-interval (operator stopped it at the
             panel) → exit without touching it further.
           - `max_broadcast_time` elapses (safety cap) → issue explicit STOP.
         Each pass re-issues PLAY, because `action=start` plays the clip once.

    Caveat on manual-stop detection: a clip shorter than `replay_interval`
    finishes on its own before we poll, and reads as a manual stop. Keep the
    deterrent clip longer than the interval (or the interval shorter than the
    clip) if you want the broadcast to run until cancelled.
    """
    if httpx is None:
        return

    cancel_signal.bind()
    try:
        async with httpx.AsyncClient(auth=httpx.BasicAuth(*auth), timeout=request_timeout) as client:
            speaker = SpeakerClient(ip, client, audio_file=audio_file, tag=tag)

            # Monotonic clock so NTP / DST adjustments can't stretch the safety
            # window. `max_broadcast_time=None` means "only a cancel or a manual
            # stop ends this", encoded as an infinite deadline to keep the loop
            # condition a plain comparison.
            deadline = math.inf if max_broadcast_time is None else time.monotonic() + max_broadcast_time
            consecutive_failures = 0
            reason = "max-broadcast-timeout"

            while time.monotonic() < deadline:
                try:
                    await speaker.play()
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_play_failures:
                        logger.error(
                            "[%s] %d consecutive PLAY failures on %s, giving up: %s",
                            tag,
                            consecutive_failures,
                            ip,
                            e,
                        )
                        try:
                            await _stop_audio(speaker, reason="play-failure-giveup")
                        except Exception:
                            pass  # already logged inside _stop_audio
                        raise
                    logger.warning(
                        "[%s] PLAY failed on %s (%d/%d), retrying in %.0fs: %s",
                        tag,
                        ip,
                        consecutive_failures,
                        max_consecutive_play_failures,
                        replay_interval,
                        e,
                    )

                if await cancel_signal.wait(timeout=replay_interval):
                    logger.info("[%s] Cancel signal for %s. Stopping broadcast.", tag, ip)
                    reason = "cancel"
                    break

                if await speaker.is_idle():
                    logger.info("[%s] Manual stop detected on %s. Broadcast finished.", tag, ip)
                    return

            if reason == "max-broadcast-timeout":
                logger.warning("[%s] Max broadcast time (%ss) reached for %s. Stopping.", tag, max_broadcast_time, ip)
            await _stop_audio(speaker, reason=reason)
    finally:
        cancel_signal.unbind()
