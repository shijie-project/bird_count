"""`AlarmDeviceGate` — turns alarm actions into "switch these devices on / off".

Shared by the speaker and smart-plug handlers, which differ only in how they
drive a device. Both consume the alarm engine's per-result annotations
(`alarm_actions`, `manual_override`) and need the same bookkeeping:

* a device is switched on by an alarm action whose level the handler
  subscribes to (`actions.<handler>.levels` in configs/alarm.json), for the
  stream's zone devices;
* several cameras share a zone's devices, so each device records who *holds*
  it — an alarm event, or a manual hijack — and is switched off only when the
  last holder lets go: the event's recovery, or the hijack being turned off;
* operator cancel-all drops every hold.

Pure bookkeeping — no threads, no I/O. The handler checks device state (only
start a READY device, only stop an ACTIVE one) under its own lock.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from alarm import DEFAULT_ACTION_LEVELS, load_config
from runtime.config import Config
from runtime.inferencer import InferenceResult


logger = logging.getLogger(__name__)

_RECOVERY = "recovery"


def subscribed_levels(config: Config, handler: str) -> frozenset[int]:
    """Levels `handler` reacts to, from configs/alarm.json `actions`.

    An unreadable config falls back to the defaults; the alarm engine reports
    the same file as broken, and with it disabled no action ever arrives anyway.
    """
    try:
        return load_config(config.envs.alarm_config_path).levels_for(handler)
    except Exception as e:
        logger.error("[%s] Could not read action levels from %s: %s", handler, config.envs.alarm_config_path, e)
        return DEFAULT_ACTION_LEVELS.get(handler, frozenset())


class AlarmDeviceGate:
    def __init__(self, stream_to_devices: Mapping[int, Iterable[str]], levels: Iterable[int]):
        self.levels = frozenset(int(level) for level in levels)
        self._devices = {int(sid): tuple(ips) for sid, ips in stream_to_devices.items()}
        self._holders: dict[str, set[str]] = {ip: set() for ips in self._devices.values() for ip in ips}

    @property
    def all_devices(self) -> list[str]:
        return sorted(self._holders)

    def update(self, result: InferenceResult) -> tuple[list[str], list[str]]:
        """Apply one result. Returns `(switch_on, switch_off)` device IPs.

        `switch_on` lists only devices that gained a *new* holder, so a device
        that stopped on its own (safety cap, unreachable) is not restarted on
        every frame of the same event — only by a fresh level or hijack.
        """
        on: list[str] = []
        off: list[str] = []
        ips = self._devices.get(int(result.stream_id), ())

        for action in result.alarm_actions:
            key = f"event:{action.event_id}"
            if action.action_type == _RECOVERY:
                off += self._release(key)
            elif action.level in self.levels:
                on += self._hold(key, ips)

        manual = f"manual:{int(result.stream_id)}"
        if result.manual_override:
            on += self._hold(manual, ips)
        else:
            off += self._release(manual)
        return _unique(on), _unique(off)

    def clear(self) -> None:
        """Cancel-all: every device loses every holder."""
        for holders in self._holders.values():
            holders.clear()

    def _hold(self, key: str, ips: Iterable[str]) -> list[str]:
        gained = []
        for ip in ips:
            if key not in self._holders[ip]:
                self._holders[ip].add(key)
                gained.append(ip)
        return gained

    def _release(self, key: str) -> list[str]:
        """Drop `key` everywhere; return the devices left with no holder."""
        freed = []
        for ip, holders in self._holders.items():
            if key in holders:
                holders.discard(key)
                if not holders:
                    freed.append(ip)
        return freed


def _unique(ips: list[str]) -> list[str]:
    return list(dict.fromkeys(ips))
