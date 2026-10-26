"""The alarm core: state machine, motion filter, config, SMS worker, evidence format."""

import csv
import json
import time
from pathlib import Path

import pytest

from birdcount.core.paths import PROJECT_ROOT
from birdcount.deploy.alarm import AlarmAction, AlarmManager, AlarmRules, CameraConfig, CountSample, load_config
from birdcount.deploy.alarm.camera_ids import mac_of, resolve_camera_id
from birdcount.deploy.alarm.evidence import COUNT_FIELDS, EvidenceStore, write_placeholder_png
from birdcount.deploy.alarm.models import MotionFilterConfig, WorkerConfig
from birdcount.deploy.alarm.motion_filter import DirectionalMotionFilter
from birdcount.deploy.alarm.sms_worker import WorkerNotifier


RAW = {"recovery_ratio": 1.0, "smoothing_seconds": 0.0}  # the bare state machine: per frame, no hysteresis


def _actions(rules: AlarmRules, series, threshold: float = 100) -> list[AlarmAction]:
    manager = AlarmManager({"cam1": CameraConfig("cam1", "Camera 1", threshold)}, rules)
    return [a for ts, n in series for a in manager.process(CountSample(timestamp=ts, camera_id="cam1", count=n))]


def test_level_escalation_and_recovery() -> None:
    rules = AlarmRules(
        alert_trigger_seconds=2, escalation_seconds=3, recovery_seconds=2, evidence_window_seconds=5, **RAW
    )
    manager = AlarmManager({"cam1": CameraConfig("cam1", "Camera 1", 100)}, rules)
    series = [(0, 90), (1, 101), (2, 101), (3, 101), (6, 101), (9, 101), (10, 101), (11, 90), (12, 90), (13, 90)]
    actions = [a for ts, n in series for a in manager.process(CountSample(timestamp=ts, camera_id="cam1", count=n))]
    assert [a.action_type for a in actions] == ["level1", "level2", "level3", "recovery"]
    assert manager.active_cameras() == set()


def test_a_count_hovering_at_the_threshold_keeps_one_event_open() -> None:
    # 5 samples/s around T=130, as axis3/B8A44FDAD555 does: dips to 122 for 40 s between breaches.
    series, t = [], 0.0
    for _ in range(4):
        for n in [135.0] * 75 + [122.0] * 200:
            series.append((t, n))
            t += 0.2
    flapping = _actions(AlarmRules(**RAW), series, threshold=130)
    assert [a.action_type for a in flapping] == ["level1", "recovery"] * 4
    held = _actions(AlarmRules(smoothing_seconds=0.0), series, threshold=130)  # clears below 0.9 x 130 = 117
    assert [a.action_type for a in held] == ["level1"]  # one open event: no recovery, no re-trigger


def test_hysteresis_still_recovers_below_the_clear_level() -> None:
    series = [(i * 0.2, 140.0) for i in range(75)] + [(15 + i * 0.2, 110.0) for i in range(200)]
    actions = _actions(AlarmRules(smoothing_seconds=0.0), series, threshold=130)
    assert [(a.action_type, a.count) for a in actions] == [("level1", 140.0), ("recovery", 110.0)]
    assert actions[1].timestamp == pytest.approx(15 + 30, abs=0.25)


def test_one_noisy_frame_breaks_neither_a_breach_nor_a_recovery() -> None:
    breach = [(i * 0.2, 99.0 if i % 10 == 5 else 104.0) for i in range(60)]  # a dip every 2 s
    assert _actions(AlarmRules(**RAW), breach) == []
    smoothed = _actions(AlarmRules(recovery_ratio=1.0), breach)
    assert [a.action_type for a in smoothed] == ["level1"] and smoothed[0].count == 104.0
    alarm = [(i * 0.2, 104.0) for i in range(60)]
    clear = [(12 + i * 0.2, 112.0 if i % 10 == 5 else 80.0) for i in range(200)]  # a spike every 2 s
    assert [a.action_type for a in _actions(AlarmRules(**RAW), alarm + clear)] == ["level1"]
    assert [a.action_type for a in _actions(AlarmRules(), alarm + clear)] == ["level1", "recovery"]


def test_a_sample_the_motion_filter_excludes_never_holds_an_event_open() -> None:
    manager = AlarmManager({"cam1": CameraConfig("cam1", "Camera 1", 100)}, AlarmRules(smoothing_seconds=0.0))
    for i in range(60):
        manager.process(CountSample(timestamp=i * 0.2, camera_id="cam1", count=120))
    walking = CountSample(timestamp=12, camera_id="cam1", count=120, decision_count=99.999, motion_excluded=True)
    manager.process(walking)
    assert manager.state_of("cam1").status == "recovering"


def test_rules_reject_a_ratio_outside_0_to_1() -> None:
    with pytest.raises(ValueError, match="recovery_ratio"):
        AlarmRules(recovery_ratio=1.2)


def _motion(xs) -> object:
    filt = DirectionalMotionFilter(
        MotionFilterConfig(
            window_seconds=3,
            min_samples=6,
            min_speed_norm_per_sec=0.03,
            min_net_displacement_norm=0.08,
            min_direction_consistency=0.85,
        )
    )
    decision = None
    for i, x in enumerate(xs):
        sample = CountSample(timestamp=i * 0.2, camera_id="cam1", count=180, centroid_x=x, centroid_y=0.5)
        decision = filt.evaluate(sample, threshold=150)
    return decision


def test_motion_filter_excludes_fast_consistent_drift() -> None:
    d = _motion([0.2 + i * 0.01 for i in range(16)])
    assert d.excluded and d.reason == "directional_motion"
    assert d.speed_norm_per_sec > 0.03 and d.net_displacement_norm > 0.08
    assert d.direction_degrees == pytest.approx(0.0, abs=5.0)
    assert d.velocity_y_norm_per_sec == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("xs", [[0.5 + (0.015 if i % 2 else -0.015) for i in range(16)], [0.5] * 16])
def test_motion_filter_keeps_jitter_and_stillness(xs) -> None:
    assert not _motion(xs).excluded


BASE = {
    "output_dir": "runs",
    "rules": {"max_level": 3},
    "cameras": [
        {"camera_id": "axis1/AAAAAAAAAAAA", "threshold": 100},
        {"camera_id": "axis1/BBBBBBBBBBBB", "threshold": 100},
    ],
}


def _load(tmp_path, **extra):
    path = tmp_path / "alarm.json"
    path.write_text(json.dumps({**BASE, **extra}), encoding="utf-8")
    return load_config(path)


def test_action_levels_default_and_override(tmp_path) -> None:
    c = _load(tmp_path)
    assert (c.levels_for("sms"), c.levels_for("speaker"), c.levels_for("smart_plug")) == ({1, 2, 3}, {1}, {1})
    c = _load(tmp_path, actions={"speaker": {"levels": [2, 3]}})
    assert c.levels_for("speaker") == {2, 3} and c.levels_for("smart_plug") == {1}
    with pytest.raises(ValueError):
        _load(tmp_path, actions={"speaker": {"levels": [4]}})


def test_extends_keeps_the_cameras(tmp_path) -> None:
    (tmp_path / "alarm.json").write_text(json.dumps(BASE), encoding="utf-8")
    debug = tmp_path / "alarm_debug.json"
    debug.write_text(json.dumps({"extends": "alarm.json", "rules": {"escalation_seconds": 20}}), encoding="utf-8")
    c = load_config(debug)
    assert c.rules.escalation_seconds == 20 and c.rules.max_level == 3 and len(c.cameras) == 2


def test_shipped_debug_config_uses_the_real_thresholds() -> None:
    real = load_config(PROJECT_ROOT / "configs" / "alarm.json")
    debug = load_config(PROJECT_ROOT / "configs" / "alarm_debug.json")
    assert {k: v.threshold for k, v in debug.cameras.items()} == {k: v.threshold for k, v in real.cameras.items()}
    assert debug.rules.escalation_seconds < real.rules.escalation_seconds


def test_camera_ids_translate_bare_macs() -> None:
    known = ["axis1/B8A44FD51C3C", "axis2/B8A44F5B4394"]
    assert resolve_camera_id(mac="b8a44fd51c3c", known=known) == "axis1/B8A44FD51C3C"
    assert resolve_camera_id(mac="AABBCCDDEEFF", known=known) is None
    assert resolve_camera_id(mac="B8A44FD51C3C", known=["B8A44FD51C3C"]) == "B8A44FD51C3C"
    assert resolve_camera_id(mac="B8A44FD51C3C", known=["axis1/B8A44FD51C3C", "axis3/B8A44FD51C3C"]) is None
    assert mac_of("axis1/B8A44FD51C3C") == "B8A44FD51C3C"


def _action() -> AlarmAction:
    return AlarmAction("level1", "cam1", "event1", time.time(), 188, 140, 1, "Level 1 pile-up alert")


def test_dry_run_sms_emulates_the_worker(tmp_path) -> None:
    snapshot = tmp_path / "snapshot.png"
    write_placeholder_png(snapshot, color=(230, 81, 0), count=188, threshold=140)
    cfg = WorkerConfig(enabled=True, dry_run=True, base_url="https://example.test", project_slug="farm-a")
    result = WorkerNotifier(cfg, mock_root=tmp_path / "mock").notify(_action(), snapshot)
    assert result["sent"] and result["dry_run"] and result["status"] == 201
    assert result["response"]["image_url"] in result["response"]["message"]["body"]
    assert Path(result["mock_log_path"]).exists() and Path(result["mock_image_path"]).exists()


def test_real_sms_needs_the_key(tmp_path) -> None:
    snapshot = tmp_path / "snapshot.png"
    write_placeholder_png(snapshot, color=(230, 81, 0), count=188, threshold=140)
    cfg = WorkerConfig(enabled=True, dry_run=False, base_url="https://example.test", project_slug="farm-a")
    result = WorkerNotifier(cfg, mock_root=tmp_path / "mock", api_key="").notify(_action(), snapshot)
    assert not result["sent"] and "FARM_SMS_API_KEY" in result["reason"]


def test_counts_csv_format_is_the_delivery_packages(tmp_path) -> None:
    store = EvidenceStore(tmp_path, AlarmRules())
    full = CountSample(
        1.5, "cam1", 120.25, None, 0.25, 0.75, 119.0, True, "directional_motion", 0.1, -0.2, 0.3, 0.4, 270.0, 0.95
    )
    bare = CountSample(2.0, "cam1", 99.0)
    for s in (full, bare):
        store.observe(s)
        store.append_event_count("cam1", "e1", s)
    rows = list(csv.reader((tmp_path / "events" / "cam1" / "e1" / "counts.csv").open(encoding="utf-8")))
    assert rows[0] == COUNT_FIELDS
    assert rows[1] == [
        "1.500",
        "1970-01-01T00:00:01.500000Z",
        "cam1",
        "120.250",
        "119.000",
        "0.250000",
        "0.750000",
        "1",
        "directional_motion",
        "0.100000",
        "-0.200000",
        "0.300000",
        "0.400000",
        "270.00",
        "0.950000",
    ]
    assert rows[2] == ["2.000", "1970-01-01T00:00:02Z", "cam1", "99.000", "", "", "", "0", "", "", "", "", "", "", ""]
