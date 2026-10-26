"""Starting, stopping and restarting the runtime from the farm backend."""

import json
import os
import subprocess
import sys
import time

import pytest

from birdcount.deploy import control
from birdcount.deploy.webui import runtime as rt

from .conftest import request


def _status(path, pid, *, started=None, stopped=None, written=None) -> None:
    now = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"written": written or now, "started": started or now, "stopped": stopped, "pid": pid, "cameras": []}
    path.write_text(json.dumps(state), encoding="utf-8")


def _settle(ctl: rt.RuntimeControl, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    while ctl.busy and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not ctl.busy, f"still {ctl.busy} ({ctl.step})"
    return ctl.state()


@pytest.fixture
def sleeper():
    """A Python process that ignores stop requests, as a runtime that hangs would."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield proc
    proc.kill()
    proc.wait(10)


class _Started:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code

    @property
    def returncode(self):
        return self.code


def test_a_stop_request_names_the_run_it_is_for(tmp_path) -> None:
    status = tmp_path / "status.json"
    stop = control.stop_file(status)
    assert stop.name == "status.stop" and not control.stop_requested(stop, 42)
    control.request_stop(status, 42)
    assert control.stop_requested(stop, 42) and not control.stop_requested(stop, 43)  # an older run's: ignored
    stop.write_text("{half", encoding="utf-8")
    assert not control.stop_requested(stop, 42)
    control.clear_stop(stop)
    control.clear_stop(stop)
    assert not stop.exists()


def test_alive_and_force_stop(sleeper) -> None:
    assert control.alive(os.getpid()) and not control.alive(None)
    control.force_stop(sleeper.pid)
    sleeper.wait(10)
    assert not control.alive(sleeper.pid)
    control.force_stop(sleeper.pid)  # already gone: nothing to do


def test_start_waits_for_the_first_heartbeat(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rt, "POLL", 0.02)
    status = tmp_path / "status.json"
    _status(status, 999_999_99, stopped=time.time() - 100, started=time.time() - 200)

    def start():
        _status(status, os.getpid())  # this process plays the runtime that came up
        return _Started()

    ctl = rt.RuntimeControl(status, start=start)
    assert ctl.run("start")["busy"] == "start"
    assert _settle(ctl) == {"busy": None, "since": ctl.since, "step": "", "error": None}
    with pytest.raises(ValueError, match="already running"):
        ctl.run("start")
    with pytest.raises(ValueError, match="unknown"):
        ctl.run("reboot")


def test_a_runtime_that_exits_at_start_is_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rt, "POLL", 0.02)
    status = tmp_path / "status.json"
    ctl = rt.RuntimeControl(status, start=lambda: _Started(code=2))
    ctl.run("start")
    assert "code 2" in _settle(ctl)["error"]
    with pytest.raises(ValueError, match="not running"):
        ctl.run("stop")
    with pytest.raises(ValueError, match="STATUS_PATH"):
        rt.RuntimeControl(None).run("start")


def test_restart_ends_a_runtime_that_does_not_stop(tmp_path, monkeypatch, sleeper) -> None:
    monkeypatch.setattr(rt, "POLL", 0.02)
    monkeypatch.setattr(rt, "STOP_TIMEOUT", 0.5)
    status = tmp_path / "status.json"
    _status(status, sleeper.pid, written=time.time() - 30)  # heartbeat stale: hung
    started = []

    def start():
        started.append(control.alive(sleeper.pid))
        _status(status, os.getpid())
        return _Started()

    ctl = rt.RuntimeControl(status, start=start)
    ctl.run("restart")
    assert _settle(ctl)["error"] is None
    assert control.stop_requested(control.stop_file(status), sleeper.pid)  # asked first
    assert started == [False]  # the old one was gone before the new one started


def test_the_page_drives_it_through_the_api(served) -> None:
    origin = {"Origin": f"http://127.0.0.1:{served.port}"}
    assert request(served, "GET", "/api/status")[1]["control"]["busy"] is None
    assert request(served, "GET", "/api/farm")[1]["status"]["control"]["error"] is None
    assert request(served, "POST", "/api/runtime", {"action": "reboot"}, origin)[0] == 409
    assert request(served, "POST", "/api/runtime", {"action": "stop"}, origin)[0] == 409  # its pid is no process
    assert request(served, "POST", "/api/runtime", {"action": "stop"}, {"Origin": "http://evil.example"})[0] == 403
