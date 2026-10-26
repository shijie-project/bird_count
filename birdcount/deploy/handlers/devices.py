"""Alarm-driven devices (zone speakers, smart plugs): one framework, small drivers.

Who switches what:
    `AlarmDeviceGate` turns decisions into "switch these devices on / off".
    A device goes on for an alarm action whose level the handler subscribes
    to, or for a manual hijack, of any camera in its zone. Several cameras
    share a zone's devices, so each device records who holds it — an alarm
    event, or a hijack — and goes off only when the last holder lets go: the
    event's recovery, or the hijack ending. Operator cancel-all drops every
    hold.

How a device is driven:
    `DeviceHandler` keeps a READY / ACTIVE state per device and runs one
    `run_lifecycle` per activation on a thread pool, so the consumer thread
    never waits on the network. A lifecycle switches the device on, then
    every `interval` refreshes it and checks it, until cancelled (switch
    off), the device reports it was switched off at its own panel (leave it),
    `max_seconds` passes (switch off), or `max_failures` consecutive steps fail
    (switch off, give up). Switching off is retried once: it is the
    safety-critical step.

A `DeviceDriver` supplies only the four device operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.base import Batch, Decision, Handler


logger = logging.getLogger(__name__)

_RECOVERY = "recovery"
_STOP_ATTEMPTS = 2
_SHUTDOWN_DRAIN_TIMEOUT = 5.0


# ----------------------------------------------------------------------------
# Who holds which device
# ----------------------------------------------------------------------------


class AlarmDeviceGate:
    def __init__(self, devices_by_camera: Mapping[int, Iterable[str]], levels: Iterable[int]):
        self.levels = frozenset(int(level) for level in levels)
        self._devices = {int(sid): tuple(ips) for sid, ips in devices_by_camera.items()}
        self._holders: dict[str, set[str]] = {ip: set() for ips in self._devices.values() for ip in ips}

    @property
    def all_devices(self) -> list[str]:
        return sorted(self._holders)

    def update(self, sid: int, decision: Decision | None, manual: bool) -> tuple[list[str], list[str]]:
        """Apply one result: (devices to switch on, devices to switch off).

        "On" lists only devices that gained a *new* holder, so a device that
        stopped on its own (safety cap, operator) is not restarted on every
        frame of the same event — only by a fresh level or hijack.
        """
        on: list[str] = []
        off: list[str] = []
        ips = self._devices.get(sid, ())
        for action in decision.actions if decision is not None else ():
            key = f"event:{action.event_id}"
            if action.action_type == _RECOVERY:
                off += self._release(key)
            elif action.level in self.levels:
                on += self._hold(key, ips)
        key = f"manual:{sid}"
        if manual:
            on += self._hold(key, ips)
        else:
            off += self._release(key)
        return list(dict.fromkeys(on)), list(dict.fromkeys(off))

    def clear(self) -> None:
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
        freed = []
        for ip, holders in self._holders.items():
            if key in holders:
                holders.discard(key)
                if not holders:
                    freed.append(ip)
        return freed


# ----------------------------------------------------------------------------
# Cancelling a lifecycle from another thread
# ----------------------------------------------------------------------------


class CancelSignal:
    """Thread-safe `request()` from any thread, awaited inside a lifecycle's loop.

    Created once per device, before any loop exists: a `request()` that comes
    before the lifecycle `bind()`s is buffered and applied on bind, so
    cancel-all has no "lifecycle not registered yet" race to handle.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None
        self._pending = False

    def request(self) -> None:
        with self._lock:
            if self._event is None:
                self._pending = True
                return
            loop, event = self._loop, self._event
        with contextlib.suppress(RuntimeError):  # loop closed meanwhile: the lifecycle is ending anyway
            loop.call_soon_threadsafe(event.set)

    def bind(self) -> None:
        """Call inside the lifecycle's running loop."""
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._event = asyncio.Event()
            if self._pending:
                self._event.set()
                self._pending = False

    def unbind(self) -> None:
        """Back to the pre-bind state, discarding any late request. Idempotent."""
        with self._lock:
            self._loop = self._event = None
            self._pending = False

    async def wait(self, timeout: float) -> bool:
        """True if cancelled within `timeout`."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False


# ----------------------------------------------------------------------------
# One activation of one device
# ----------------------------------------------------------------------------


class DeviceDriver(Protocol):
    """One activation's view of a device. An async context: `__aenter__` opens
    the connection, `__aexit__` closes it."""

    ip: str

    async def __aenter__(self) -> DeviceDriver: ...
    async def __aexit__(self, *exc) -> None: ...
    async def start(self) -> None: ...  # switch on (raise on failure)
    async def refresh(self) -> None: ...  # keep it going, e.g. replay a clip (raise on failure)
    async def is_on(self) -> bool | None: ...  # False: switched off at the device; None: unknown
    async def stop(self) -> None: ...  # switch off (raise on failure)


@dataclass(frozen=True)
class LifecycleTiming:
    interval: float  # between refresh / check passes
    max_seconds: float | None  # safety cap; None = until cancelled or switched off by hand
    max_failures: int = 5  # consecutive failed passes before giving up


async def _stop(driver: DeviceDriver, reason: str, tag: str) -> None:
    for attempt in range(1, _STOP_ATTEMPTS + 1):
        try:
            await driver.stop()
            logger.info("[%s] %s switched off (%s).", tag, driver.ip, reason)
            return
        except Exception as e:
            if attempt == _STOP_ATTEMPTS:
                logger.error("[%s] Could not switch %s off (%s): %s", tag, driver.ip, reason, e)
                raise
            logger.warning("[%s] Switching %s off failed (%s), retrying: %s", tag, driver.ip, reason, e)


async def run_lifecycle(driver: DeviceDriver, cancel: CancelSignal, timing: LifecycleTiming, tag: str) -> str:
    """Drive one activation; returns why it ended: cancel / manual / timeout."""
    cancel.bind()
    try:
        async with driver:
            await driver.start()
            logger.info("[%s] %s switched on.", tag, driver.ip)
            deadline = math.inf if timing.max_seconds is None else time.monotonic() + timing.max_seconds
            failures = 0
            while time.monotonic() < deadline:
                failed = None
                try:
                    await driver.refresh()
                except Exception as e:
                    failed = e
                if await cancel.wait(timing.interval):
                    await _stop(driver, "cancel", tag)
                    return "cancel"
                if failed is None:
                    try:
                        on = await driver.is_on()
                    except Exception as e:
                        failed, on = e, None
                    if on is False:
                        logger.info("[%s] %s was switched off at the device.", tag, driver.ip)
                        return "manual"
                if failed is None:
                    failures = 0
                    continue
                failures += 1
                if failures >= timing.max_failures:
                    logger.error("[%s] %d failures in a row on %s, giving up: %s", tag, failures, driver.ip, failed)
                    with contextlib.suppress(Exception):
                        await _stop(driver, "giving up", tag)
                    raise failed
                logger.warning("[%s] %s: %s (%d/%d)", tag, driver.ip, failed, failures, timing.max_failures)
            logger.warning("[%s] %s reached its %.0f s safety cap.", tag, driver.ip, timing.max_seconds)
            await _stop(driver, "safety cap", tag)
            return "timeout"
    finally:
        cancel.unbind()


# ----------------------------------------------------------------------------
# The handler
# ----------------------------------------------------------------------------

_READY, _ACTIVE = "READY", "ACTIVE"


class DeviceHandler(Handler):
    """Switches a kind of device (speakers, plugs) for alarms and manual hijacks."""

    def __init__(
        self,
        name: str,
        devices_by_camera: Mapping[int, Iterable[str]],
        levels: Iterable[int],
        driver_factory: Callable[[str], DeviceDriver],
        timing: LifecycleTiming,
        *,
        audit: AuditLog = NULL_AUDIT,
    ):
        super().__init__(audit)
        self.name = name
        self.gate = AlarmDeviceGate(devices_by_camera, levels)
        self.driver_factory = driver_factory
        self.timing = timing
        self.states = dict.fromkeys(self.gate.all_devices, _READY)
        self._signals = {ip: CancelSignal() for ip in self.states}
        self._lock = threading.Lock()  # states: consumer thread, pool threads, GUI thread
        self._pool: ThreadPoolExecutor | None = None
        logger.info("[%s] %d device(s), on alarm level(s) %s.", name, len(self.states), sorted(self.gate.levels))

    def start(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.states)), thread_name_prefix=self.name)
        self.audit.log(f"{self.name.lower()}.devices", devices=sorted(self.states))

    def stop(self) -> None:
        self.cancel_all()
        deadline = time.monotonic() + _SHUTDOWN_DRAIN_TIMEOUT
        while self.active_devices() and time.monotonic() < deadline:
            time.sleep(0.1)
        if left := self.active_devices():
            logger.warning("[%s] Still ON after %.0f s: %s", self.name, _SHUTDOWN_DRAIN_TIMEOUT, sorted(left))
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def handle(self, batch: Batch) -> None:
        if self._pool is None:
            return
        sids, counts = batch.records["sid"].tolist(), batch.records["count"].tolist()
        for sid, count, decision in zip(sids, counts, batch.decisions, strict=True):
            switch_on, switch_off = self.gate.update(sid, decision, sid in batch.manual)
            for ip in switch_on:
                with self._lock:
                    if self.states[ip] != _READY:
                        continue
                    self.states[ip] = _ACTIVE
                self.audit.log(
                    "device.activate",
                    handler=self.name,
                    ip=ip,
                    stream_id=sid,
                    count=count,
                    manual=sid in batch.manual,
                )
                self._pool.submit(self._run, ip)
            for ip in switch_off:
                # Only an ACTIVE device has a lifecycle to stop; signalling a
                # READY one would pre-arm it and kill its next activation.
                with self._lock:
                    if self.states[ip] != _ACTIVE:
                        continue
                    self._signals[ip].request()
                self.audit.log("device.release", handler=self.name, ip=ip, stream_id=sid)

    def _run(self, ip: str) -> None:
        signal = self._signals[ip]
        error = None
        try:
            reason = asyncio.run(run_lifecycle(self.driver_factory(ip), signal, self.timing, self.name))
        except Exception as e:
            reason, error = "failed", str(e)
            logger.error("[%s] Lifecycle of %s failed: %s", self.name, ip, e)
        finally:
            with self._lock:
                self.states[ip] = _READY
                # Discard a cancel that landed after the lifecycle unbound, or
                # it would kill the next activation at birth.
                signal.unbind()
        self.audit.log("device.deactivate", handler=self.name, ip=ip, reason=reason, error=error)

    def cancel_all(self) -> None:
        self.gate.clear()
        with self._lock:
            active = [ip for ip, st in self.states.items() if st == _ACTIVE]
        for ip in active:
            self._signals[ip].request()
        if active:
            logger.info("[%s] Cancel-all: switching off %s.", self.name, active)
            self.audit.log("handler.cancel_all", handler=self.name, devices=active)

    def active_devices(self) -> set[str]:
        with self._lock:
            return {ip for ip, st in self.states.items() if st == _ACTIVE}
