"""Zone speakers: replay the deterrent clip while an alarm holds them.

Talks to the speaker's `/cgi-bin` audio API. `action=start` plays the clip
once, so the lifecycle's periodic `refresh` replays it; the speaker reporting
"stopped"/"idle" between replays means someone stopped it at the device.
Keep the clip longer than the replay interval, or a clip that simply ended
reads as a manual stop.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.devices import DeviceHandler, LifecycleTiming


logger = logging.getLogger(__name__)

TIMING = LifecycleTiming(interval=10.0, max_seconds=3600.0, max_failures=5)
REQUEST_TIMEOUT = 5.0
_IDLE_TOKENS = ("stopped", "idle")


class SpeakerDriver:
    def __init__(self, ip: str, auth: tuple[str, str], clip: str):
        self.ip = ip
        self._auth = auth
        self._clip = clip
        self._client = None

    async def __aenter__(self) -> SpeakerDriver:
        import httpx

        self._client = httpx.AsyncClient(auth=httpx.BasicAuth(*self._auth), timeout=REQUEST_TIMEOUT)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def _get(self, endpoint: str, params: dict | None = None):
        response = await self._client.get(f"http://{self.ip}/cgi-bin/{endpoint}", params=params)
        response.raise_for_status()
        return response

    async def start(self) -> None:
        """Nothing to do: the first `refresh` plays the clip."""

    async def refresh(self) -> None:
        await self._get("audio_play", {"name": self._clip, "action": "start", "time": 1})

    async def is_on(self) -> bool | None:
        """Unknown (None) when the status cannot be read: dropping a deterrent
        on a parse failure is the worse error."""
        try:
            body = (await self._get("audio_get_status")).text.lower()
        except Exception as e:
            logger.debug("Status of speaker %s unavailable: %s", self.ip, e)
            return None
        return not any(token in body for token in _IDLE_TOKENS)

    async def stop(self) -> None:
        await self._get("audio_play", {"name": self._clip, "action": "stop", "time": 1})


def speaker_handler(
    speakers_by_camera: Mapping[int, Iterable[str]],
    levels: Iterable[int],
    *,
    username: str,
    password: str,
    clip: str,
    audit: AuditLog = NULL_AUDIT,
) -> DeviceHandler | None:
    """None (logged) when httpx is not installed."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        logger.error("Speakers need the 'httpx' package; speaker alarms are off.")
        return None
    return DeviceHandler(
        "Speaker",
        speakers_by_camera,
        levels,
        lambda ip: SpeakerDriver(ip, (username, password), clip),
        TIMING,
        audit=audit,
    )
