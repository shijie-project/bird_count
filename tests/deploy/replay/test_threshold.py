import csv
import json
from pathlib import Path

import numpy as np
import pytest

from birdcount.deploy import countlog
from birdcount.deploy.replay.layout import RunLayout
from birdcount.deploy.replay.threshold import (
    Episode,
    _set_day,
    _try_one,
    parse_thresholds,
    score,
    verdicts,
)

from .conftest import DAY, MAC, SIZE, THRESHOLD, ScriptedPredictor, replay


def ramp(i: int) -> float:
    """150 for 4 s, 200 for 4 s, then 10: over 140 for 8 s, over 180 for 4 s."""
    return 150.0 if i < 20 else 200.0 if i < 40 else 10.0


@pytest.fixture
def run(out, clips, config):
    layout = RunLayout(out / "run")
    replay(layout, clips, ScriptedPredictor(ramp), config)
    return layout


def test_thresholds_to_try() -> None:
    assert parse_thresholds("100..140") == [100, 110, 120, 130, 140]
    assert parse_thresholds("100..110:5, 135") == [100, 105, 110, 135]
    with pytest.raises(ValueError):
        parse_thresholds("140..100")


def test_deciding_again_at_the_threshold_counted_at_gives_the_same_alarms(run, config_path) -> None:
    _, per_day = _try_one((str(run.root), str(config_path), MAC, THRESHOLD, [DAY]))
    counted = [json.loads(line) for line in (run.evidence / "events.jsonl").read_text().splitlines()]
    starts = [a["timestamp"] for a in counted if a["action_type"] == "level1"]
    ends = [a["timestamp"] for a in counted if a["action_type"] == "recovery"]
    assert per_day[DAY] == [Episode(starts[0], ends[0])]


def test_a_higher_threshold_alarms_later_or_not_at_all(run, config_path) -> None:
    (_, at_140), (_, at_180), (_, at_250) = (
        _try_one((str(run.root), str(config_path), MAC, t, [DAY])) for t in (140, 180, 250)
    )
    assert at_180[DAY][0].start > at_140[DAY][0].start
    assert at_250[DAY] == []


def test_a_verdict_given_at_one_threshold_scores_the_alarms_of_another(run, config_path) -> None:
    actions = [json.loads(line) for line in (run.evidence / "events.jsonl").read_text().splitlines()]
    picture = run.evidence / next(a for a in actions if a["action_type"] == "level1")["snapshot_path"]
    key = countlog.ALARM_PREFIX + picture.relative_to(run.evidence).as_posix()
    with (run.count_log / DAY / countlog.LABELS_FILE).open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["snapshot", "label", "labeled_at"], [key, "pileup", "now"]])
    judged = verdicts(run, MAC)
    assert [label for _, label in judged] == ["pileup"]
    _, at_180 = _try_one((str(run.root), str(config_path), MAC, 180, [DAY]))
    assert score(at_180[DAY][0], judged) == "real"
    assert score(Episode(0, 1), judged) == "unchecked"


def test_setting_a_threshold_redoes_the_days_alarm_with_pictures(run, config_path) -> None:
    record = _set_day((str(run.root), str(config_path), MAC, 180.0, DAY, list(SIZE)))
    run.set_threshold(MAC, 180.0)
    run.finish(MAC, DAY, record)

    assert record["threshold"] == 180.0
    assert record["alarms"]["level1"] == 1
    assert record["seconds_over"]["1.0"] == 4
    folder = run.evidence_of(MAC, DAY, 180.0)
    actions = [json.loads(line) for line in (run.evidence / "events.jsonl").read_text().splitlines()]
    assert actions and all(a["threshold"] == 180.0 for a in actions)  # the index shows the threshold set
    assert all(folder in Path(a["snapshot_path"]).parents for a in actions)
    assert all(a["snapshot_path"].endswith(".jpg") for a in actions)  # the frame, decoded again
    assert run.evidence_of(MAC, DAY, THRESHOLD).is_dir()  # the counting threshold's evidence (and verdicts) stay

    levels = np.load(run.samples_of(MAC, DAY))["level"]
    with (run.count_log / DAY / f"{MAC}.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert max(int(r["alarm_level"]) for r in rows) == levels.max() > 0
    meta = json.loads((run.count_log / DAY / "meta.json").read_text(encoding="utf-8"))
    assert meta["thresholds"] == {MAC: 180.0}
    assert run.thresholds() == {MAC: 180.0}
