"""Device routing (gate + handler) and the shared device lifecycle, on fake drivers."""

import asyncio
import time

import pytest

from birdcount.deploy.alarm import AlarmAction
from birdcount.deploy.handlers.base import Batch, Decision
from birdcount.deploy.handlers.devices import (
    AlarmDeviceGate,
    CancelSignal,
    DeviceHandler,
    LifecycleTiming,
    run_lifecycle,
)

from .conftest import records


def _action(kind: str, event: str, level=None) -> AlarmAction:
    return AlarmAction(kind, "axis1/A", event, 0.0, 150.0, 100.0, level, "")


def _decision(*actions: AlarmAction) -> Decision:
    level = max((a.level or 0 for a in actions), default=0)
    return Decision("axis1/A", None, tuple(actions), level, actions[0].event_id if actions else None, False)


@pytest.fixture
def gate():
    # Cameras 0 and 1 share a speaker; camera 2 has none.
    return AlarmDeviceGate({0: ["10.0.0.9"], 1: ["10.0.0.9"], 2: []}, levels=[1])


def test_subscribed_level_on_recovery_off(gate) -> None:
    assert gate.update(0, _decision(_action("level1", "e1", 1)), False) == (["10.0.0.9"], [])
    assert gate.update(0, _decision(), False) == ([], [])
    assert gate.update(0, _decision(_action("recovery", "e1")), False) == ([], ["10.0.0.9"])


def test_unsubscribed_level_is_ignored(gate) -> None:
    assert gate.update(0, _decision(_action("level2", "e1", 2)), False) == ([], [])


def test_shared_device_stays_on_until_the_last_event_recovers(gate) -> None:
    gate.update(0, _decision(_action("level1", "e1", 1)), False)
    gate.update(1, _decision(_action("level1", "e2", 1)), False)
    assert gate.update(0, _decision(_action("recovery", "e1")), False) == ([], [])
    assert gate.update(1, _decision(_action("recovery", "e2")), False) == ([], ["10.0.0.9"])


def test_manual_hijack_holds_until_released(gate) -> None:
    assert gate.update(0, None, True) == (["10.0.0.9"], [])
    assert gate.update(0, None, True) == ([], [])
    assert gate.update(0, None, False) == ([], ["10.0.0.9"])


def test_clear_drops_every_hold(gate) -> None:
    gate.update(0, _decision(_action("level1", "e1", 1)), False)
    gate.clear()
    assert gate.update(0, _decision(_action("recovery", "e1")), False) == ([], [])


# ----------------------------------------------------------------------------


class FakeDriver:
    """Scriptable device: records calls, can fail or report a manual switch-off."""

    def __init__(self, ip, *, off_after=None, fail_refresh=0, fail_check=False, fail_stop=0):
        self.ip = ip
        self.calls: list[str] = []
        self.off_after = off_after  # is_on() turns False after this many checks
        self.fail_refresh = fail_refresh
        self.fail_check = fail_check
        self.fail_stop = fail_stop
        self.checks = 0

    async def __aenter__(self):
        self.calls.append("open")
        return self

    async def __aexit__(self, *exc):
        self.calls.append("close")

    async def start(self):
        self.calls.append("start")

    async def refresh(self):
        self.calls.append("refresh")
        if self.fail_refresh:
            self.fail_refresh -= 1
            raise OSError("refresh failed")

    async def is_on(self):
        self.checks += 1
        if self.fail_check:
            raise OSError("poll failed")
        return not (self.off_after is not None and self.checks > self.off_after)

    async def stop(self):
        self.calls.append("stop")
        if self.fail_stop:
            self.fail_stop -= 1
            raise OSError("stop failed")


FAST = LifecycleTiming(interval=0.01, max_seconds=None, max_failures=3)


def _run(driver, timing=FAST, cancel_after=None):
    signal = CancelSignal()

    async def main():
        if cancel_after is not None:
            asyncio.get_running_loop().call_later(cancel_after, signal.request)
        return await run_lifecycle(driver, signal, timing, "test")

    return asyncio.run(main())


def test_cancel_switches_off() -> None:
    d = FakeDriver("ip")
    assert _run(d, cancel_after=0.05) == "cancel"
    assert d.calls[:2] == ["open", "start"]
    assert d.calls[-2:] == ["stop", "close"]


def test_a_cancel_before_the_lifecycle_binds_is_kept() -> None:
    d, signal = FakeDriver("ip"), CancelSignal()
    signal.request()  # e.g. cancel-all racing the pool picking the job up
    assert asyncio.run(run_lifecycle(d, signal, FAST, "test")) == "cancel"


def test_switched_off_at_the_device_ends_without_stop() -> None:
    d = FakeDriver("ip", off_after=2)
    assert _run(d) == "manual"
    assert "stop" not in d.calls


def test_safety_cap_switches_off() -> None:
    d = FakeDriver("ip")
    assert _run(d, LifecycleTiming(interval=0.01, max_seconds=0.05, max_failures=3)) == "timeout"
    assert d.calls[-2:] == ["stop", "close"]


def test_transient_failures_are_retried() -> None:
    d = FakeDriver("ip", fail_refresh=2)  # fewer than max_failures
    assert _run(d, cancel_after=0.1) == "cancel"


def test_persistent_failures_switch_off_and_give_up() -> None:
    d = FakeDriver("ip", fail_check=True)
    with pytest.raises(OSError, match="poll failed"):
        _run(d)
    assert "stop" in d.calls


def test_switching_off_is_retried_once() -> None:
    d = FakeDriver("ip", fail_stop=1)
    assert _run(d, cancel_after=0.03) == "cancel"
    assert d.calls.count("stop") == 2


# ----------------------------------------------------------------------------


def _handler(levels=(1,)):
    drivers: dict[str, FakeDriver] = {}

    def factory(ip):
        drivers[ip] = FakeDriver(ip)
        return drivers[ip]

    return DeviceHandler("Speaker", {0: ["10.0.0.9"]}, levels, factory, FAST), drivers


def _batch(*actions, manual=False):
    return Batch(records(0, 0.0, 150.0), [_decision(*actions)], frozenset({0}) if manual else frozenset())


def _until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        time.sleep(0.01)
    assert predicate()


def test_handler_on_at_level1_off_at_recovery() -> None:
    h, drivers = _handler()
    h.start()
    h.handle(_batch(_action("level1", "e1", 1)))
    _until(lambda: "10.0.0.9" in drivers and "start" in drivers["10.0.0.9"].calls)
    assert h.active_devices() == {"10.0.0.9"}
    h.handle(_batch(_action("level2", "e1", 2)))  # same event: not restarted
    assert len(drivers) == 1
    h.handle(_batch(_action("recovery", "e1")))
    _until(lambda: h.active_devices() == set())
    assert drivers["10.0.0.9"].calls[-2:] == ["stop", "close"]
    h.stop()


def test_recovery_does_not_pre_arm_a_ready_device() -> None:
    """A cancel signalled to a READY device would kill its next activation at birth."""
    h, drivers = _handler(levels=(2,))
    h.start()
    h.handle(_batch(_action("level1", "e1", 1)))  # not subscribed: stays READY
    h.handle(_batch(_action("recovery", "e1")))
    h.handle(_batch(manual=True))  # the next activation must run normally
    _until(lambda: h.active_devices() == {"10.0.0.9"})
    time.sleep(0.05)
    assert "stop" not in drivers["10.0.0.9"].calls
    h.stop()
    assert drivers["10.0.0.9"].calls[-2:] == ["stop", "close"]


def test_cancel_all_then_a_new_hold() -> None:
    h, _ = _handler()
    h.start()
    h.handle(_batch(manual=True))
    _until(lambda: h.active_devices() == {"10.0.0.9"})
    h.cancel_all()
    _until(lambda: h.active_devices() == set())
    h.handle(_batch(manual=True))  # cancel-all dropped the hold: the hijack takes it again
    _until(lambda: h.active_devices() == {"10.0.0.9"})
    h.stop()
    assert h.active_devices() == set()
