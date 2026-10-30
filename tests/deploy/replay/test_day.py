import csv
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from birdcount.deploy.replay.layout import RunLayout, RunMismatch
from birdcount.deploy.replay.recordings import Clip

from .conftest import DAY, MAC, THRESHOLD, ScriptedPredictor, piled_up_for_two_clips, replay


def test_a_day_is_counted_on_the_recordings_clock(out, clips, config) -> None:
    layout = RunLayout(out / "run")
    record = replay(layout, clips, ScriptedPredictor(piled_up_for_two_clips), config)

    samples = np.load(layout.samples_of(MAC, DAY))
    t0 = datetime(2026, 3, 12, 10, 0, 0).timestamp()
    np.testing.assert_allclose(samples["t"], t0 + np.arange(90) * 0.2)  # every 2nd frame of three 6 s clips
    assert samples["clip"].tolist() == [0] * 30 + [1] * 30 + [2] * 30
    assert samples["frame"].tolist() == list(range(0, 60, 2)) * 3
    assert samples["count"][:60].tolist() == [200.0] * 60
    assert samples["level"].max() == 3 and samples["level"][-1] == 0  # escalated, then recovered
    assert list(samples["clips"]) == [str(c.path) for c in clips]

    assert record["samples"] == 90
    assert record["alarms"] == {"level1": 1, "level2": 1, "level3": 1, "recovery": 1}
    assert record["seconds_over"] == {"0.6": 12, "0.8": 12, "1.0": 12}
    assert [c["samples"] for c in record["clips"]] == [30, 30, 30]
    assert [c["max_level"] for c in record["clips"]] == [2, 3, 3]  # L2 at 5 s, L3 at 8 s, clear at 14 s

    with (layout.count_log / DAY / f"{MAC}.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18  # one row a second
    assert rows[0]["time_local"] == "2026-03-12 10:00:00" and rows[0]["frames"] == "5"

    actions = [json.loads(line) for line in (layout.evidence / "events.jsonl").read_text().splitlines()]
    assert [a["action_type"] for a in actions] == ["level1", "level2", "level3", "recovery"]
    assert all(Path(a["snapshot_path"]).suffix == ".jpg" for a in actions)  # the frame itself, not a placeholder
    assert all(Path(a["snapshot_path"]).is_relative_to(layout.evidence) for a in actions)


def test_a_clip_overlapping_the_one_before_drops_the_seconds_it_repeats(out, clips, config) -> None:
    layout = RunLayout(out / "run")
    overlapping = [clips[0], Clip(clips[1].path, datetime(2026, 3, 12, 10, 0, 3))]  # starts 3 s early
    record = replay(layout, overlapping, ScriptedPredictor(lambda i: 10.0), config)
    t = np.load(layout.samples_of(MAC, DAY))["t"]
    assert (np.diff(t) > 0).all()
    assert record["clips"][1]["dropped"] == 15
    assert record["samples"] == 45


def test_an_unreadable_clip_is_recorded_and_skipped(out, clips, config) -> None:
    layout = RunLayout(out / "run")
    broken = out / "rec" / "20260312_100006_BAD0.mkv"
    broken.write_bytes(b"not a video")
    record = replay(
        layout, [clips[0], Clip(broken, clips[1].start), clips[2]], ScriptedPredictor(lambda i: 10.0), config
    )
    assert [c["samples"] for c in record["clips"]] == [30, 0, 30]
    assert record["clips"][1]["error"]


def test_a_stopped_day_is_not_marked_done(out, clips, config) -> None:
    layout = RunLayout(out / "run")
    stop = threading.Event()
    stop.set()
    assert replay(layout, clips, ScriptedPredictor(lambda i: 10.0), config, stop=stop) is None
    assert not layout.done(MAC, DAY)


def test_counting_a_day_again_replaces_it(out, clips, config) -> None:
    layout = RunLayout(out / "run")
    replay(layout, clips, ScriptedPredictor(piled_up_for_two_clips), config)
    replay(layout, clips, ScriptedPredictor(piled_up_for_two_clips), config)
    actions = (layout.evidence / "events.jsonl").read_text().splitlines()
    assert len(actions) == 4
    with (layout.count_log / DAY / f"{MAC}.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 18
    assert len(layout.records()) == 1


def test_the_days_meta_names_every_camera_replayed_into_it(out, clips, config) -> None:
    layout = RunLayout(out / "run")
    record = replay(layout, clips, ScriptedPredictor(lambda i: 10.0), config)
    other = {**record, "mac": "E82725003277", "threshold": 120.0, "meta": {"streams": {"0": "x"}}}
    layout.record_of("E82725003277", DAY).parent.mkdir(parents=True)
    layout.record_of("E82725003277", DAY).write_text(json.dumps(other), encoding="utf-8")
    replay(layout, clips, ScriptedPredictor(lambda i: 10.0), config)
    meta = json.loads((layout.count_log / DAY / "meta.json").read_text(encoding="utf-8"))
    assert meta["thresholds"] == {MAC: THRESHOLD, "E82725003277": 120.0}
    assert meta["replayed"] == [MAC, "E82725003277"]
    assert "streams" not in meta


def test_a_run_continues_only_under_the_same_model_and_rates(out) -> None:
    layout = RunLayout(out / "run")
    about = {"model_sha256": "abc", "frame_size": [1280, 720], "fps": 5, "mask_dir": "m"}
    layout.open(about)
    layout.open({**about, "mask_dir": "other"})  # masks are recorded per day, not fixed per run
    with pytest.raises(RunMismatch, match="fps"):
        layout.open({**about, "fps": 10})
