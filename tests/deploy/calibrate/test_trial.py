"""Trying thresholds on the farm's count log, scored by the verdicts on its pictures."""

import csv
import json
from datetime import datetime

from birdcount.deploy import countlog
from birdcount.deploy.calibrate import trial


MAC = "B8A44FD51C3C"
DAY = "2026-11-03"
CONFIG = {
    "output_dir": "evidence",
    "rules": {"alert_trigger_seconds": 10.0, "escalation_seconds": 120.0, "recovery_seconds": 10.0, "max_level": 3},
    "motion_filter": {"enabled": False},
    "cameras": [{"camera_id": f"axis1/{MAC}", "threshold": 110}],
}
# Two crowded stretches: 120 birds for a minute (alarms at 110), then 95 for a minute (only below 95).
STRETCHES = [(0, 100, 50.0), (100, 160, 120.0), (160, 260, 50.0), (260, 320, 95.0), (320, 400, 50.0)]


def _count_log(tmp_path):
    root = tmp_path / "count_log"
    (root / DAY).mkdir(parents=True)
    t0 = int(datetime(2026, 11, 3, 9, 0, 0).timestamp())
    with (root / DAY / f"{MAC}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=countlog.ROW_FIELDS)
        w.writeheader()
        for a, b, n in STRETCHES:
            for s in range(a, b):
                w.writerow({"time_local": "", "timestamp": t0 + s, "frames": 3, "count_mean": n, "count_max": n,
                            "count_min": n, "centroid_x": 0.5, "centroid_y": "", "alarm_level": int(n > 110),
                            "motion_excluded_frames": 0, "snapshot": ""})  # fmt: skip
    labels = [
        (countlog.snapshot_path(MAC, t0 + 130, "high", 120), "pileup"),  # in the first stretch
        (countlog.snapshot_path(MAC, t0 + 290, "high", 95), "pileup"),  # in the second
        (countlog.snapshot_path(MAC, t0 + 50, "bin", 50), "normal"),  # quiet: no alarm holds it
        (countlog.snapshot_path("B8A44F5B4329", t0 + 290, "high", 95), "normal"),  # another camera
    ]
    with (root / DAY / countlog.LABELS_FILE).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snapshot", "label", "labeled_at"])
        w.writerows([rel, label, ""] for rel, label in labels)
    config = tmp_path / "alarm.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    return root, config


def test_thresholds_are_decided_again_and_scored_by_the_pictures(tmp_path, capsys) -> None:
    root, config = _count_log(tmp_path)
    day = trial.read_day(root, DAY, MAC)
    assert day.hours == 400 / 3600 and trial.logged_alarms(day) == 1
    judged = trial.verdicts(root, tmp_path / "evidence", MAC, [DAY])
    assert sorted(label for _, label in judged) == ["normal", "pileup", "pileup"]

    assert trial.run(root, tmp_path / "evidence", config, MAC, [90.0, 130.0], [DAY], jobs=1) == 0
    out = capsys.readouterr().out
    assert "decided again on per-second counts: 1 Level 1 alarm(s); the runtime raised 1" in out
    rows = {
        line.split()[0]: line.split() for line in out.splitlines() if line.split()[:1] in (["90"], ["110"], ["130"])
    }
    # T, Level 1, per day, alarming, m, real, false, unsure, unchecked, missed
    assert rows["90"][1] == "2" and rows["90"][5] == "2" and rows["90"][9] == "0"
    assert rows["110"][1] == "1" and rows["110"][5] == "1" and rows["110"][9] == "1" and rows["110"][-1] == "now"
    assert rows["130"][1] == "0" and rows["130"][9] == "2"
