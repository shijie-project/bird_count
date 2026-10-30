import csv

import numpy as np

from birdcount.core.frames import read_image
from birdcount.deploy import countlog
from birdcount.deploy.replay.layout import RunLayout
from birdcount.deploy.replay.snapshots import Rules, pick, smoothed, snapshot_camera

from .conftest import DAY, MAC, ScriptedPredictor, replay


def test_the_smoothed_count_is_the_alarms_2_s_median() -> None:
    count = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 100, 10, 10], np.float32)  # one noisy frame
    assert smoothed(count, 5).max() == 10
    assert smoothed(count[:3], 5).tolist() == [10, 10, 10]  # shorter than the window: as counted


def test_bins_are_pictured_apart_and_peaks_once_each() -> None:
    t = np.arange(0, 6 * 3600, 0.2)  # six hours at 5 samples a second
    count = np.full(len(t), 45.0, np.float32)
    count[(t > 3600) & (t < 3660)] = 130.0  # one busy minute
    rules = Rules(floor=40, per_bin=3, gap=20 * 60, peak_floor=80, peak_gap=10 * 60)
    picks = pick(t, count, smoothed(count, 5), rules, seed=1)
    peaks = [p for p in picks if p.why == "peak"]
    assert len(peaks) == 1 and 3600 < t[peaks[0].i] < 3660
    in_40s = [t[p.i] for p in picks if p.why == "bin" and count[p.i] == 45]
    assert len(in_40s) == 3 and np.diff(sorted(in_40s)).min() >= 20 * 60
    assert [p.i for p in picks] == sorted(p.i for p in picks)
    assert picks == pick(t, count, smoothed(count, 5), rules, seed=1)  # the same moments again


def test_pictures_land_in_the_data_folder_and_the_runs_count_log(out, clips, config) -> None:
    layout = RunLayout(out / "run")
    replay(layout, clips, ScriptedPredictor(lambda i: 40.0 + i), config)  # 40 .. 129 over 18 s
    layout.open({"model_sha256": "abc", "frame_size": [64, 48], "fps": 5})
    rules = Rules(floor=60, per_bin=1, gap=0.0, peak_floor=1000, peak_gap=60)
    n = snapshot_camera(layout, MAC, out / "data", rules, jobs=2, copy_ahead=0)

    with (out / "data" / MAC / "index.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert n == len(rows) == 7  # one per bin from 60 to 120
    assert [int(float(r["count"])) // 10 * 10 for r in rows] == [60, 70, 80, 90, 100, 110, 120]
    samples = np.load(layout.samples_of(MAC, DAY))
    for r in rows:
        picture = read_image(out / "data" / r["file"])
        assert picture.shape == (48, 64, 3)  # the frame as counted, not the review size
        i = int(np.nonzero(np.isclose(samples["t"], float(r["timestamp"]), rtol=0, atol=1e-3))[0][0])
        assert int(r["frame"]) == samples["frame"][i]

    small = sorted(p.name for p in (layout.count_log / DAY / "snapshots" / MAC).iterdir())
    assert small == sorted(r["file"].rsplit("/", 1)[1] for r in rows)
    with (layout.count_log / DAY / f"{MAC}.csv").open(encoding="utf-8") as f:
        pointed = [row["snapshot"] for row in csv.DictReader(f) if row["snapshot"]]
    assert len(pointed) == len({int(float(r["timestamp"])) for r in rows})  # one picture a second at most
    assert all(countlog.parse_snapshot(DAY, p) and p.rsplit("/", 1)[1] in small for p in pointed)
