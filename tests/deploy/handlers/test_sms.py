"""Engine + SMS handler end to end, asserting on what lands on disk."""

import json
import time
from pathlib import Path

from birdcount.deploy.handlers.engine import SIMULATED_COUNT_FACTOR, AlarmEngine
from birdcount.deploy.handlers.sms import SmsHandler

from .conftest import CAMERA_ID, ESCALATION, MAC, THRESHOLD, records, run, write_config


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sms(cfg, out, **kw) -> SmsHandler:
    h = SmsHandler(cfg, output_dir=out, **kw)
    h.start()
    return h


def test_escalation_and_recovery_write_evidence_and_sms(alarm_config, ring, out) -> None:
    engine, sms = AlarmEngine(alarm_config, [MAC]), _sms(alarm_config, out)
    run(engine, [sms], ring, ESCALATION, time.time())
    sms.stop()
    events = _jsonl(out / "events.jsonl")
    assert [e["action_type"] for e in events] == ["level1", "level2", "level3", "recovery"]
    for e in events:  # real frames, not the placeholder PNG
        assert Path(e["snapshot_path"]).suffix == ".jpg" and Path(e["snapshot_path"]).stat().st_size > 0
    notes = _jsonl(out / "notifications.jsonl")
    assert len(notes) == 4
    assert all(n["notification"]["sent"] and n["notification"]["dry_run"] for n in notes)
    rows = next((out / "events").rglob("counts.csv")).read_text(encoding="utf-8").splitlines()
    assert len(rows) > 1 and "decision_count" in rows[0]


def test_below_threshold_never_alarms(alarm_config, ring, out) -> None:
    engine, sms = AlarmEngine(alarm_config, [MAC]), _sms(alarm_config, out)
    run(engine, [sms], ring, [(i, THRESHOLD - 1) for i in range(20)], time.time())
    sms.stop()
    assert not (out / "events.jsonl").exists()
    assert sms.active_devices() == set()


def test_an_uncovered_camera_never_alarms(alarm_config, ring, out) -> None:
    engine, sms = AlarmEngine(alarm_config, [None]), _sms(alarm_config, out)
    batches = run(engine, [sms], ring, [(i, 500) for i in range(20)], time.time())
    sms.stop()
    assert all(b.decisions == [None] for b in batches)
    assert not (out / "events.jsonl").exists()


def test_cancel_all_drops_the_open_event(alarm_config, ring, out) -> None:
    engine, sms = AlarmEngine(alarm_config, [MAC]), _sms(alarm_config, out)
    run(engine, [sms], ring, [(0, 200), (1, 200), (2, 200), (3, 200)], time.time())
    assert engine.active_cameras() == {CAMERA_ID}
    assert sms.active_devices() == {CAMERA_ID}
    engine.reset_all()
    sms.cancel_all()
    assert engine.active_cameras() == set()
    assert sms.active_devices() == set()
    sms.stop()


def test_sms_levels_limit_texts_not_evidence(tmp_path, ring, out) -> None:
    cfg = write_config(tmp_path, actions={"sms": {"levels": [2]}})
    engine, sms = AlarmEngine(cfg, [MAC]), _sms(cfg, out)
    run(engine, [sms], ring, ESCALATION, time.time())
    sms.stop()
    assert [e["action_type"] for e in _jsonl(out / "events.jsonl")] == ["level1", "level2", "level3", "recovery"]
    assert [n["action_type"] for n in _jsonl(out / "notifications.jsonl")] == ["level2", "recovery"]


def test_engine_levels_and_alert(alarm_config, ring) -> None:
    engine = AlarmEngine(alarm_config, [MAC])
    batches = run(engine, [], ring, ESCALATION, time.time())
    seen = [(b.decisions[0].level, b.alert(0), [a.action_type for a in b.decisions[0].actions]) for b in batches]
    assert seen[3] == (1, True, ["level1"])
    assert seen[5] == (3, True, ["level3"])
    assert seen[-1] == (0, False, ["recovery"])
    decisions, manual = engine.evaluate(records(0, time.time() + 100, 0.0), {0})
    assert manual == {0} and decisions[0].level == 0  # a plain hijack opens no event


def test_debug_hijack_simulates_a_full_event(alarm_config, ring, out) -> None:
    engine = AlarmEngine(alarm_config, [MAC], simulate_manual=True)
    sms = _sms(alarm_config, out)
    series = [(step / 2, 50.0) for step in range(32)]  # the real count stays low; hijacked for t < 12
    batches = run(engine, [sms], ring, series, time.time(), hijacked=lambda t: {0} if t < 12 else set())
    sms.stop()
    assert all(b.manual == frozenset() for b in batches)  # simulated, not a device-only override
    events = _jsonl(out / "events.jsonl")
    assert [e["action_type"] for e in events] == ["level1", "level2", "level3", "recovery"]
    assert events[0]["count"] == SIMULATED_COUNT_FACTOR * THRESHOLD
    assert "manual_simulation" in next((out / "events").rglob("counts.csv")).read_text(encoding="utf-8")


def test_hijack_does_not_simulate_unless_asked(alarm_config) -> None:
    engine = AlarmEngine(alarm_config, [MAC], simulate_manual=False)
    decisions, manual = engine.evaluate(records(0, time.time(), 50.0), {0})
    assert manual == {0}
    assert not decisions[0].simulated
    assert engine.active_cameras() == set()
