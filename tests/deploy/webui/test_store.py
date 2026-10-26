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


def test_the_label_queue_asks_about_disagreements_first() -> None:
    def pic(key, count, source="count_log", t=0):
        return {"key": key, "count": count, "source": source, "t": t}

    pictures = [
        pic("near", 98),
        pic("far", 40),
        pic("alarm:a.jpg", 130, "alarm", 5),
        pic("over", 140),  # over the threshold, rated calm
        pic("under", 50),  # well under it, rated crammed
    ]
    levels = {"over": 2, "under": 5, "near": 4, "far": 2}
    queue = store.label_queue(pictures, 100, levels, labelled=set())
    assert queue == {"over": "over", "under": "under", "alarm:a.jpg": "alarm"}
    no_ratings = store.label_queue(pictures, 100, {}, labelled=set())
    assert no_ratings == {"alarm:a.jpg": "alarm", "near": "near", "over": "near"}
    # labelled pictures keep their place: working through the queue never refills it
    done = store.label_queue(pictures, 100, levels, labelled={"far", "near", "over"})
    assert done == {"far": "near", "near": "near", "over": "over"}
