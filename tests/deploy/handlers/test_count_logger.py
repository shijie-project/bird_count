import csv
import json
from datetime import datetime

import pytest

from birdcount.deploy import countlog
from birdcount.deploy.handlers import count_logger
from birdcount.deploy.handlers.base import Batch
from birdcount.deploy.handlers.count_logger import CountLogger
from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.settings import parse_hours

from .conftest import MAC, records, write_config


def test_parse_hours() -> None:
    start, end = parse_hours("06:00-20:00")
    assert (start.hour, end.hour) == (6, 20)
    with pytest.raises(ValueError):
        parse_hours("20:00-06:00")


def _logger(tmp_path):
    cfg = write_config(tmp_path, cameras=[{"camera_id": f"axis1/{MAC}", "threshold": 100}])
    model = tmp_path / "model.pth"
    model.write_bytes(b"weights")
    engine = AlarmEngine(cfg, [MAC])
    return CountLogger(tmp_path / "log", "06:00-20:00", engine, [MAC], meta={"fps": 5}, model_path=model), engine


def _feed(log, engine, ring, t0, counts, fps=5):
    for i, n in enumerate(counts):
        recs = records(0, t0 + i / fps, n, cx=0.25, cy=0.75)
        decisions, manual = engine.evaluate(recs, set())
        log.handle(Batch(recs, decisions, manual, ring))


def test_rows_meta_and_snapshots(tmp_path, ring) -> None:
    log, engine = _logger(tmp_path)
    log.start()
    # 1 s at 40 birds (under the floor, 60 % of T=100), 2 s at 70, then 1 s at 90 (>= 80 % of T: "high")
    _feed(log, engine, ring, datetime(2026, 10, 25, 7, 0, 0).timestamp(), [40] * 5 + [70] * 10 + [90] * 5)
    log.stop()
    day = tmp_path / "log" / "2026-10-25"
    rows = list(csv.DictReader((day / f"{MAC}.csv").open(encoding="utf-8")))
    assert [r["frames"] for r in rows] == ["5", "5", "5", "5"]
    assert [r["count_max"] for r in rows] == ["40.00", "70.00", "70.00", "90.00"]
    assert rows[0]["centroid_x"] == "0.2500"
    assert rows[0]["time_local"] == "2026-10-25 07:00:00"
    assert rows[0]["snapshot"] == ""  # under the floor
    assert rows[1]["snapshot"].endswith("_bin_70.jpg")
    assert rows[2]["snapshot"] == ""  # its bin already has a picture this minute
    assert rows[3]["snapshot"].endswith("_high_90.jpg")
    for r in (rows[1], rows[3]):
        assert (day / r["snapshot"]).exists()
    meta = json.loads((day / "meta.json").read_text(encoding="utf-8"))
    assert meta["thresholds"] == {MAC: 100.0} and meta["snapshot_floors"] == {MAC: 60}
    assert len(meta["model_sha256"]) == 64
    assert meta["fps"] == 5


def test_snapshots_cover_each_count_bin_a_few_times_a_day(tmp_path, ring) -> None:
    log, engine = _logger(tmp_path)
    log.start()
    t0 = datetime(2026, 10, 25, 7, 0, 0).timestamp()
    for minute in range(120):  # two hours: 60 birds, then 75, alternating every minute
        _feed(log, engine, ring, t0 + 60 * minute, [60 if minute % 2 == 0 else 75] * 5)
    log.stop()
    shots = sorted(p.name for p in (tmp_path / "log" / "2026-10-25" / "snapshots" / MAC).iterdir())
    assert sum("_bin_60" in s for s in shots) == countlog.SAMPLES_PER_BIN
    assert sum("_bin_75" in s for s in shots) == countlog.SAMPLES_PER_BIN
    first, second = sorted(s[:6] for s in shots if "_bin_60" in s)[:2]
    assert (int(second[2:4]) - int(first[2:4])) % 60 >= count_logger.SAMPLE_GAP_SEC // 60


def test_outside_the_window_writes_nothing(tmp_path, ring) -> None:
    log, engine = _logger(tmp_path)
    log.start()
    _feed(log, engine, ring, datetime(2026, 10, 25, 21, 30, 0).timestamp(), [150] * 20)
    log.stop()
    assert not (tmp_path / "log").exists()
