import json
import time
from types import SimpleNamespace

import numpy as np

from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.handlers.status import WARN_AFTER_FAILURES, StatusFile

from .conftest import CAMERA_ID, ESCALATION, MAC, THRESHOLD, run


def _status(tmp_path, ring):
    ring.status = np.array([(1,), (0,)], dtype=[("online", "u1")])
    cams = [
        {"mac": MAC, "label": "clip a", "zone": "Zone_01", "threshold": THRESHOLD},
        {"mac": None, "label": "clip b", "zone": None, "threshold": None},
    ]
    return StatusFile(tmp_path / "live" / "status.json", ring, cams, runtime={"sms": "dry run"})


def test_the_file_follows_counts_and_alarm_levels(tmp_path, ring, alarm_config) -> None:
    status = _status(tmp_path, ring)
    status.start()
    engine = AlarmEngine(alarm_config, [MAC, None])
    t0 = time.time() - 20
    run(engine, [status], ring, ESCALATION[:5], t0)  # through Level 1, into Level 2
    status.tick(now=t0 + 100)
    state = json.loads(status.path.read_text(encoding="utf-8"))
    a, b = state["cameras"]
    assert (a["count"], a["level"], a["online"], a["threshold"]) == (200.0, 2, True, THRESHOLD)
    assert a["alarm_since"] == t0 + 3 and a["event_id"].startswith(CAMERA_ID.replace("/", "_"))
    assert (b["count"], b["level"], b["online"]) == (None, 0, False)
    assert state["runtime"] == {"sms": "dry run"} and state["stopped"] is None
    run(engine, [status], ring, [(20, 50), (25, 50)], t0)
    status.stop()
    a = json.loads(status.path.read_text(encoding="utf-8"))["cameras"][0]
    assert (a["level"], a["alarm_since"]) == (0, None)
    assert json.loads(status.path.read_text(encoding="utf-8"))["stopped"] is not None


def test_ticks_write_at_most_once_a_second(tmp_path, ring) -> None:
    status = _status(tmp_path, ring)
    status.start()
    first = json.loads(status.path.read_text(encoding="utf-8"))["written"]
    status.tick(now=first + 0.5)
    assert json.loads(status.path.read_text(encoding="utf-8"))["written"] == first
    status.tick(now=first + 1.5)
    assert json.loads(status.path.read_text(encoding="utf-8"))["written"] == first + 1.5
    assert not list(status.path.parent.glob("*.tmp"))


def test_an_unwritable_place_only_warns(tmp_path, caplog) -> None:
    ring = SimpleNamespace(status=np.array([(1,)], dtype=[("online", "u1")]))
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    status = StatusFile(blocker / "status.json", ring, [{"mac": MAC, "label": "a", "zone": None, "threshold": 1}], {})
    for _ in range(WARN_AFTER_FAILURES - 1):
        status.write()  # one failed rename now and then is a reader in the way: no warning yet
    assert not any("Cannot write the status file" in r.message for r in caplog.records)
    for _ in range(3):
        status.write()
    assert sum("Cannot write the status file" in r.message for r in caplog.records) == 1


def test_a_failed_last_write_warns_at_once(tmp_path, caplog) -> None:
    ring = SimpleNamespace(status=np.array([(1,)], dtype=[("online", "u1")]))
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    status = StatusFile(blocker / "status.json", ring, [{"mac": MAC, "label": "a", "zone": None, "threshold": 1}], {})
    status.stop()
    assert sum("Cannot write the status file" in r.message for r in caplog.records) == 1
