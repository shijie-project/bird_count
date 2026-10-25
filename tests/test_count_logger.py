"""CountLoggerHandler: per-second rows, the logging window, and snapshots."""

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from runtime.handlers.count_logger import CountLoggerHandler
from runtime.handlers.count_logger._handler import parse_hours
from runtime.inferencer import BatchInferenceResult, InferenceResult


MAC = "B8A44FD52B7F"


class _StubSHM:
    def __init__(self):
        self.frames = np.full((1, 2, 72, 128, 3), 120, dtype=np.uint8)


def _config(tmp: Path) -> SimpleNamespace:
    alarm = tmp / "alarm.json"
    alarm.write_text(
        json.dumps({"output_dir": "runs", "cameras": [{"camera_id": f"axis1/{MAC}", "threshold": 100}]}),
        encoding="utf-8",
    )
    model = tmp / "model.pth"
    model.write_bytes(b"weights")
    return SimpleNamespace(
        num_streams=1,
        sid_to_mac={0: MAC},
        envs=SimpleNamespace(
            enable_count_logger=True,
            count_log_dir=str(tmp / "log"),
            count_log_hours="06:00-20:00",
            alarm_config_path=str(alarm),
            model_path=str(model),
            image_width=1280,
            image_height=720,
            mask_dir="masks",
            fps=5,
            audit_log_path="",
        ),
    )


def _feed(handler, t0: float, counts, fps=5):
    shm = _StubSHM()
    for i, count in enumerate(counts):
        result = InferenceResult(0, 0, i, t0 + i / fps, float(count), 0.0, centroid_x=0.25, centroid_y=0.75)
        handler.handle_batch(BatchInferenceResult([result]), shm)


class CountLoggerTest(unittest.TestCase):
    def test_parse_hours(self):
        start, end = parse_hours("06:00-20:00")
        self.assertEqual((start.hour, end.hour), (6, 20))
        with self.assertRaises(ValueError):
            parse_hours("20:00-06:00")

    def test_rows_meta_and_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            handler = CountLoggerHandler(_config(tmp), shm_config=None)
            handler.start()
            t0 = datetime(2026, 10, 25, 7, 0, 0).timestamp()
            # 2 s at 50 birds, then 1 s at 90 (>= 80 % of T=100 -> a "high" snapshot)
            _feed(handler, t0, [50] * 10 + [90] * 5)
            handler.stop()

            day = tmp / "log" / "2026-10-25"
            rows = list(csv.DictReader((day / f"{MAC}.csv").open(encoding="utf-8")))
            self.assertEqual([r["frames"] for r in rows], ["5", "5", "5"])
            self.assertEqual([r["count_max"] for r in rows], ["50.00", "50.00", "90.00"])
            self.assertEqual(rows[0]["centroid_x"], "0.2500")
            self.assertEqual(rows[0]["time_local"], "2026-10-25 07:00:00")

            # first second: routine snapshot; third: the high one
            self.assertTrue(rows[0]["snapshot"].endswith("_routine_50.jpg"))
            self.assertTrue(rows[2]["snapshot"].endswith("_high_90.jpg"))
            for r in (rows[0], rows[2]):
                self.assertTrue((day / r["snapshot"]).exists())

            meta = json.loads((day / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["thresholds"], {MAC: 100.0})
            self.assertEqual(len(meta["model_sha256"]), 64)

    def test_outside_hours_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            handler = CountLoggerHandler(_config(tmp), shm_config=None)
            handler.start()
            _feed(handler, datetime(2026, 10, 25, 21, 30, 0).timestamp(), [150] * 20)
            handler.stop()
            self.assertFalse((tmp / "log").exists())

    def test_high_snapshots_are_rate_limited(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            handler = CountLoggerHandler(_config(tmp), shm_config=None)
            handler.start()
            _feed(handler, datetime(2026, 10, 25, 9, 0, 0).timestamp(), [120] * (5 * 90))  # 90 s over
            handler.stop()
            snaps = sorted((tmp / "log" / "2026-10-25" / "snapshots" / MAC).glob("*_high_*.jpg"))
            self.assertEqual(len(snaps), 2)  # t=0 and t=60


if __name__ == "__main__":
    unittest.main()
