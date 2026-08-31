import unittest

from alarm.models import AlarmRules, CameraConfig, CountSample
from alarm.state_machine import AlarmManager


class AlarmStateMachineTest(unittest.TestCase):
    def test_level_escalation_and_recovery(self):
        camera = CameraConfig("cam1", "Camera 1", 100)
        rules = AlarmRules(
            alert_trigger_seconds=2,
            escalation_seconds=3,
            recovery_seconds=2,
            evidence_window_seconds=5,
            max_level=3,
        )
        manager = AlarmManager({"cam1": camera}, rules)

        actions = []
        values = [
            (0, 90),
            (1, 101),
            (2, 101),
            (3, 101),
            (6, 101),
            (9, 101),
            (10, 101),
            (11, 90),
            (12, 90),
            (13, 90),
        ]
        for ts, count in values:
            actions.extend(manager.process(CountSample(timestamp=ts, camera_id="cam1", count=count)))

        self.assertEqual([a.action_type for a in actions], ["level1", "level2", "level3", "recovery"])


if __name__ == "__main__":
    unittest.main()
