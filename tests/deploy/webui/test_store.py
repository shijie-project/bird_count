"""Reading the runtime's files: placement, live status, count logs (incrementally), labels, evidence."""

import json
import time

from birdcount.deploy import countlog
from birdcount.deploy.webui import store

from .conftest import DAY, LOOSE, PLACED, T0, row, write_rows


def test_cameras_sit_in_their_zone_or_under_their_recorder(farm) -> None:
    by_mac = {c.mac: c for c in farm.cameras}
    assert (by_mac[PLACED].zone, by_mac[PLACED].group) == ("Zone_01_Right_Front", "Zone_01_Right_Front")
    assert (by_mac[LOOSE].zone, by_mac[LOOSE].group) == (None, "axis2")
    zones = {z["name"]: z for z in farm.zones}
    front = zones["Zone_01_Right_Front"]
    assert (front["side"], front["row"], front["macs"], front["cameras_listed"], front["speakers"]) == (
        "right", "front", [PLACED], 2, 1,
    )  # fmt: skip
    assert (zones["Zone_05_Right_Outer"]["row"], zones["axis2"]["kind"], zones["axis2"]["macs"]) == (
        "outer",
        "group",
        [LOOSE],
    )


def test_status_is_running_only_while_the_heartbeat_moves(farm) -> None:
    path = farm.status_path
    assert store.read_status(path)["state"] == "running"
    assert store.read_status(path, now=time.time() + 60)["state"] == "gone"
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "stopped": time.time()}), encoding="utf-8")
    assert store.read_status(path)["state"] == "stopped"
    assert store.read_status(path.with_name("none.json"))["state"] == "never"
    assert store.read_status(None)["state"] == "disabled"
    path.write_text("{half", encoding="utf-8")
    assert store.read_status(path)["state"] == "unreadable"


def test_a_growing_count_log_is_read_from_where_it_stopped(farm) -> None:
    s = farm.logs.series(DAY, PLACED)
    assert len(s.ts) == 120 and s.snapshots[0] == (T0 + 10, f"snapshots/{PLACED}/070010_bin_40.jpg")
    path = farm.logs.day_dir(DAY) / f"{PLACED}.csv"
    offset = s.offset
    write_rows(path, [row(T0 + 120, 50)], header=False)
    with path.open("a", encoding="utf-8") as f:
        f.write("2026-10-25 07:02:01,")  # a row the writer has not finished
    s = farm.logs.series(DAY, PLACED)
    assert len(s.ts) == 121 and s.offset > offset and s.ts[-1] == T0 + 120
    assert farm.logs.series(DAY, LOOSE) is None


def test_a_day_summarised_and_folded(farm) -> None:
    s = farm.logs.series(DAY, PLACED)
    summary = store.summarize(s, 100)
    assert summary["periods"] == [[T0 + 70, T0 + 120, 1]]
    assert (summary["alarm_seconds"], summary["above_threshold_seconds"], summary["max_level"]) == (50, 60, 1)
    assert (summary["count_now"], summary["count_max"], summary["level_now"]) == (110.0, 112.0, 1)
    assert summary["peak"] <= summary["count_max"] and T0 <= summary["peak_at"] <= T0 + 119  # a 1-s mean, not a frame
    c = store.curve(s, max_points=40)
    assert c["step"] == 3 and len(c["t"]) == 40
    assert (c["low"][0], c["high"][-1], c["level"][23], c["level"][24]) == (38.0, 112.0, 1, 1)
    assert countlog.parse_snapshot(DAY, f"snapshots/{PLACED}/070100_high_110.jpg") == {
        "path": f"snapshots/{PLACED}/070100_high_110.jpg", "t": T0 + 60, "kind": "high", "count": 110, "mac": PLACED,
    }  # fmt: skip


def test_the_last_verdict_on_a_snapshot_wins(farm) -> None:
    day_dir = farm.logs.day_dir(DAY)
    labels = store.Labels()
    shot = f"snapshots/{PLACED}/070100_high_110.jpg"
    labels.write(day_dir, shot, "normal")
    labels.write(day_dir, shot, "pileup")
    assert labels.read(day_dir)[shot]["label"] == "pileup"
    labels.write(day_dir, shot, "")
    assert shot not in labels.read(day_dir)
    try:
        labels.write(day_dir, shot, "maybe")
    except ValueError as e:
        assert "pileup, normal, unsure" in str(e)


def test_alarm_actions_by_day_with_their_pictures(farm) -> None:
    (action,) = farm.evidence.of_day(DAY)
    assert action["mac"] == PLACED and action["image"].endswith(f"level1_{T0 + 70}.jpg")
    assert farm.evidence.of_day("2026-10-26") == []
    path = farm.evidence.root / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**json.loads(path.read_text().splitlines()[0]), "action_type": "recovery"}) + "\n")
    assert [a["action_type"] for a in farm.evidence.actions()] == ["level1", "recovery"]


def test_alarm_actions_become_events_with_their_pictures(farm) -> None:
    path = farm.evidence.root / "events.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**first, "action_type": "level2", "level": 2, "timestamp": T0 + 250}) + "\n")
        f.write(json.dumps({**first, "action_type": "recovery", "level": None, "timestamp": T0 + 400}) + "\n")
    (event,) = store.alarm_events(farm.evidence.actions())
    assert (event["t"], event["level"], event["end"]) == (int(T0 + 70.2), 2, T0 + 400)
    assert [p["type"] for p in event["pictures"]] == ["level1", "level2", "recovery"]
    assert event["key"] == f"alarm:{event['pictures'][0]['path']}"  # the verdict goes on the first picture


def test_evidence_rewritten_whole_is_read_again(tmp_path) -> None:
    """A replay rewrites events.jsonl (sorted, other camera-days merged in); the runtime only appends."""
    ev = store.Evidence(tmp_path)
    path = tmp_path / "events.jsonl"

    def action(t: float) -> str:
        return json.dumps({"camera_id": f"axis1/{PLACED}", "timestamp": t, "action_type": "level1"}) + "\n"

    path.write_text(action(T0 + 10), encoding="utf-8")
    assert [a["timestamp"] for a in ev.actions()] == [T0 + 10]
    tmp = tmp_path / "events.jsonl.tmp"
    tmp.write_text(action(T0) + action(T0 + 10) + action(T0 + 20), encoding="utf-8")
    tmp.replace(path)  # longer, and an earlier action in front: an offset into the old file means nothing
    assert [a["timestamp"] for a in ev.actions()] == [T0, T0 + 10, T0 + 20]
