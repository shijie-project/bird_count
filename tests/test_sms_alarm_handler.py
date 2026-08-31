"""End-to-end test for `runtime.handlers.sms_alarm`, without a GPU or cameras.

Drives the handler with synthetic `InferenceResult` batches and a stub SHM
client, and asserts on what actually lands on disk: the Level 1/2/3 + recovery
sequence, real JPEG snapshots taken from the frame buffer, and one dry-run SMS
payload per action.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from runtime.handlers.sms_alarm import SmsAlarmHandler
from runtime.inferencer import BatchInferenceResult, InferenceResult


CAMERA_ID = "axis1/B8A44FD52B7F"
MAC = "B8A44FD52B7F"  # what topology.yaml / a clip filename carries
THRESHOLD = 140

# Compressed rules so the test runs in milliseconds of simulated time.
_ALARM_CONFIG = {
    "output_dir": "runs",
    "sample_fps": 5.0,
    "rules": {
        "alert_trigger_seconds": 2.0,
        "escalation_seconds": 3.0,
        "recovery_seconds": 2.0,
        "evidence_window_seconds": 5.0,
        "max_level": 3,
    },
    "motion_filter": {"enabled": False},
    "worker": {
        "enabled": True,
        "dry_run": True,
        "base_url": "https://example.test",
        "project_slug": "test-farm",
        "mock_recipients": ["+61400000000"],
    },
    "cameras": [{"camera_id": CAMERA_ID, "name": "test camera", "threshold": THRESHOLD}],
}


class _StubSHM:
    """Stands in for `runtime.shared_memory.SharedMemory`.

    Only `.frames[sid, buffer_idx]` is exercised — the handler resolves frames
    by hand and never touches buffer metadata.
    """

    def __init__(self, num_streams: int = 1, num_buffers: int = 2, h: int = 8, w: int = 8):
        self.frames = np.zeros((num_streams, num_buffers, h, w, 3), dtype=np.uint8)
        self.frames[...] = 200  # non-black, so a failed imwrite is distinguishable


def _make_config(tmp: Path) -> SimpleNamespace:
    """Minimal `runtime.config.Config` stand-in covering what the handler reads."""
    config_path = tmp / "alarm.json"
    config_path.write_text(json.dumps(_ALARM_CONFIG), encoding="utf-8")
    return SimpleNamespace(
        num_streams=1,
        sid_to_mac={0: MAC},
        sid_to_source={0: "rtsp://stub"},
        sid_to_ip={0: "10.0.0.1"},
        envs=SimpleNamespace(
            enable_sms_alarm=True,
            sms_alarm_real_worker=False,
            alarm_config_path=str(config_path),
            alarm_output_dir=str(tmp / "out"),
            audit_log_path="",  # disables the audit file
        ),
    )


class SmsAlarmHandlerTest(unittest.TestCase):
    def _run_sequence(self, handler, shm, series, t0):
        for offset, count in series:
            batch = BatchInferenceResult(
                results=[
                    InferenceResult(
                        stream_id=0,
                        buffer_idx=0,
                        frame_idx=0,
                        timestamp=t0 + offset,
                        count=float(count),
                        latency=0.0,
                    )
                ]
            )
            handler.handle_batch(batch, shm)

    def test_full_escalation_and_recovery_writes_evidence_and_sms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out"
            handler = SmsAlarmHandler(_make_config(tmp), shm_config=None)
            handler.start()
            self.assertTrue(handler._enabled, "handler disabled itself during start()")

            shm = _StubSHM()
            t0 = time.time()
            # over threshold long enough for L1, then L2, then L3, then clear
            series = [(0, 90), (1, 200), (2, 200), (3, 200), (6, 200), (9, 200), (10, 200)]
            series += [(11, 50), (12, 50), (13, 50)]
            self._run_sequence(handler, shm, series, t0)
            handler.stop()  # drains the evidence thread + notify pool

            events = [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [e["action_type"] for e in events],
                ["level1", "level2", "level3", "recovery"],
            )

            # Every action carried a real frame, not the stdlib placeholder PNG.
            for event in events:
                snapshot = Path(event["snapshot_path"])
                self.assertTrue(snapshot.exists(), f"missing snapshot for {event['action_type']}")
                self.assertEqual(snapshot.suffix, ".jpg")
                self.assertGreater(snapshot.stat().st_size, 0)

            notifications = [
                json.loads(line) for line in (out / "notifications.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(notifications), 4)
            for record in notifications:
                self.assertTrue(record["notification"]["sent"])
                self.assertTrue(record["notification"]["dry_run"])

            # Per-sample counts.csv is written for the open event only.
            csvs = list((out / "events").rglob("counts.csv"))
            self.assertEqual(len(csvs), 1)
            rows = csvs[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertGreater(len(rows), 1)
            self.assertIn("decision_count", rows[0])

    def test_below_threshold_never_alarms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out"
            handler = SmsAlarmHandler(_make_config(tmp), shm_config=None)
            handler.start()
            self._run_sequence(handler, _StubSHM(), [(i, THRESHOLD - 1) for i in range(20)], time.time())
            handler.stop()

            self.assertFalse((out / "events.jsonl").exists())
            self.assertEqual(handler.get_active_devices(), set())

    def test_unmapped_stream_disables_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = _make_config(tmp)
            # A stream with no resolvable MAC at all (a camera-mode stream
            # whose zone lists no camera_ids).
            config.sid_to_mac = {0: None}
            config.sid_to_source = {0: "rtsp://192.168.0.9/stream"}
            config.sid_to_ip = {0: "192.168.0.9"}

            handler = SmsAlarmHandler(config, shm_config=None)
            handler.start()
            self.assertFalse(handler._enabled)
            # Still safe to drive — it just does nothing.
            self._run_sequence(handler, _StubSHM(), [(i, 500) for i in range(20)], time.time())
            handler.stop()

    def test_cancel_all_drops_the_open_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            handler = SmsAlarmHandler(_make_config(tmp), shm_config=None)
            handler.start()

            t0 = time.time()
            self._run_sequence(handler, _StubSHM(), [(0, 200), (1, 200), (2, 200), (3, 200)], t0)
            self.assertEqual(handler.get_active_devices(), {CAMERA_ID})

            handler.cancel_all()
            self.assertEqual(handler.get_active_devices(), set())
            handler.stop()


if __name__ == "__main__":
    unittest.main()
