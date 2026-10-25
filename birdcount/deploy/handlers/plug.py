"""Zone smart plugs (TP-Link Tapo P100): on while an alarm holds them.

One `tapo.ApiClient` is shared by every activation: its Rust backend runs its
own runtime and is safe to use from several threads and event loops at once.
It is created on first use, inside the process that drives the plugs.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterable, Mapping

from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.devices import DeviceHandler, LifecycleTiming


logger = logging.getLogger(__name__)

TIMING = LifecycleTiming(interval=5.0, max_seconds=None, max_failures=5)
ON_OFF_TIMEOUT = 10.0
POLL_TIMEOUT = 5.0


class PlugDriver:
    def __init__(self, ip: str, client):
        self.ip = ip
        self._client = client
        self._device = None

    async def __aenter__(self) -> PlugDriver:
        self._device = await self._client.p100(self.ip)
        return self

    async def __aexit__(self, *exc) -> None:
        self._device = None

    async def start(self) -> None:
        await asyncio.wait_for(self._device.on(), timeout=ON_OFF_TIMEOUT)

    async def refresh(self) -> None:
        """A plug stays on by itself."""

    async def is_on(self) -> bool | None:
        info = await asyncio.wait_for(self._device.get_device_info(), timeout=POLL_TIMEOUT)
        return bool(info.device_on)

    async def stop(self) -> None:
        await asyncio.wait_for(self._device.off(), timeout=ON_OFF_TIMEOUT)


class _SharedClient:
    """Builds the tapo client on first use (it cannot be pickled across spawn)."""

    def __init__(self, email: str, password: str):
        self._credentials = (email, password)
        self._client = None
        self._lock = threading.Lock()

    def __call__(self, ip: str) -> PlugDriver:
        with self._lock:
            if self._client is None:
                from tapo import ApiClient

                self._client = ApiClient(*self._credentials)
        return PlugDriver(ip, self._client)


def plug_handler(
    plugs_by_camera: Mapping[int, Iterable[str]],
    levels: Iterable[int],
    *,
    email: str,
    password: str,
    audit: AuditLog = NULL_AUDIT,
) -> DeviceHandler | None:
    """None (logged) when the tapo package is not installed."""
    try:
        import tapo  # noqa: F401
    except ImportError:
        logger.error("Smart plugs need the 'tapo' package; plug alarms are off.")
        return None
    return DeviceHandler("SmartPlug", plugs_by_camera, levels, _SharedClient(email, password), TIMING, audit=audit)
