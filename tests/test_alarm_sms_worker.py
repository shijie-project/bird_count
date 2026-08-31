import tempfile
import time
import unittest
from pathlib import Path

from alarm.evidence import write_placeholder_png
from alarm.models import AlarmAction, WorkerConfig
from alarm.sms_worker import WorkerNotifier


class WorkerNotifierTest(unittest.TestCase):
    def test_dry_run_emulates_worker_response_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.png"
            write_placeholder_png(snapshot, color=(230, 81, 0), count=188, threshold=140)

            action = AlarmAction(
                action_type="level1",
                camera_id="cam1",
                event_id="event1",
                timestamp=time.time(),
                count=188,
                threshold=140,
                level=1,
                message="Level 1 pile-up alert",
            )
            config = WorkerConfig(
                enabled=True,
                dry_run=True,
                base_url="https://example.test",
                project_slug="farm-a",
                mock_recipients=("+61400000000",),
            )

            result = WorkerNotifier(config, mock_root=root / "mock_worker").notify(action, snapshot)

            self.assertTrue(result["sent"])
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["status"], 201)
            self.assertIn("image_url", result["response"])
            self.assertIn(result["response"]["image_url"], result["response"]["message"]["body"])
            self.assertTrue(Path(result["mock_log_path"]).exists())
            self.assertTrue(Path(result["mock_image_path"]).exists())

    def test_real_mode_requires_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.png"
            write_placeholder_png(snapshot, color=(230, 81, 0), count=188, threshold=140)
            action = AlarmAction(
                action_type="level1",
                camera_id="cam1",
                event_id="event1",
                timestamp=time.time(),
                count=188,
                threshold=140,
                level=1,
                message="Level 1 pile-up alert",
            )
            config = WorkerConfig(
                enabled=True,
                dry_run=False,
                base_url="https://example.test",
                project_slug="farm-a",
                api_key_env="CHICKEN_ALARM_TEST_MISSING_KEY",
            )

            result = WorkerNotifier(config, mock_root=root / "mock_worker").notify(action, snapshot)

            self.assertFalse(result["sent"])
            self.assertFalse(result["dry_run"])
            self.assertIn("missing API key", result["reason"])


if __name__ == "__main__":
    unittest.main()
