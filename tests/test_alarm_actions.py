"""Alarm level -> device routing: `actions` config, `AlarmDeviceGate`, and the
speaker / smart-plug handlers acting on the alarm engine's annotations.

No devices are contacted: lifecycles are captured by a fake executor and cancel
requests by a fake signal, so the tests assert on *what would be switched*.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from alarm import AlarmAction, load_config
from runtime.handlers._alarm_gate import AlarmDeviceGate
from runtime.handlers.smart_plug import SmartPlugHandler
from runtime.handlers.speaker import SpeakerHandler
from runtime.inferencer import InferenceResult


_BASE_CONFIG = {
    "output_dir": "runs",
    "rules": {"max_level": 3},
    "cameras": [
        {"camera_id": "axis1/AAAAAAAAAAAA", "threshold": 100},
        {"camera_id": "axis1/BBBBBBBBBBBB", "threshold": 100},
    ],
}


def _action(kind: str, event: str, level=None) -> AlarmAction:
    return AlarmAction(kind, "axis1/AAAAAAAAAAAA", event, 0.0, 150.0, 100.0, level, "")


def _result(sid: int, *actions: AlarmAction, manual: bool = False) -> InferenceResult:
    result = InferenceResult(sid, 0, 0, 0.0, 150.0, 0.0)
    result.alarm_actions = tuple(actions)
    result.manual_override = manual
    result.alarm_level = max((a.level or 0 for a in actions), default=0)
    return result


class ActionLevelsConfigTest(unittest.TestCase):
    def _load(self, **extra):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alarm.json"
            path.write_text(json.dumps({**_BASE_CONFIG, **extra}), encoding="utf-8")
            return load_config(path)

    def test_defaults_without_actions_block(self):
        config = self._load()
        self.assertEqual(config.levels_for("sms"), {1, 2, 3})
        self.assertEqual(config.levels_for("speaker"), {1})
        self.assertEqual(config.levels_for("smart_plug"), {1})

    def test_actions_block_overrides_per_handler(self):
        config = self._load(actions={"speaker": {"levels": [2, 3]}})
        self.assertEqual(config.levels_for("speaker"), {2, 3})
        self.assertEqual(config.levels_for("smart_plug"), {1})

    def test_level_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            self._load(actions={"speaker": {"levels": [4]}})


class AlarmDeviceGateTest(unittest.TestCase):
    def setUp(self):
        # Two cameras (sid 0, 1) in one zone sharing a speaker; sid 2 has none.
        self.gate = AlarmDeviceGate({0: ["10.0.0.9"], 1: ["10.0.0.9"], 2: []}, levels=[1])

    def test_subscribed_level_switches_on_and_recovery_off(self):
        self.assertEqual(self.gate.update(_result(0, _action("level1", "e1", 1))), (["10.0.0.9"], []))
        self.assertEqual(self.gate.update(_result(0)), ([], []))
        self.assertEqual(self.gate.update(_result(0, _action("recovery", "e1"))), ([], ["10.0.0.9"]))

    def test_unsubscribed_level_is_ignored(self):
        self.assertEqual(self.gate.update(_result(0, _action("level2", "e1", 2))), ([], []))

    def test_shared_device_stays_on_until_last_event_recovers(self):
        self.gate.update(_result(0, _action("level1", "e1", 1)))
        self.gate.update(_result(1, _action("level1", "e2", 1)))
        self.assertEqual(self.gate.update(_result(0, _action("recovery", "e1"))), ([], []))
        self.assertEqual(self.gate.update(_result(1, _action("recovery", "e2"))), ([], ["10.0.0.9"]))

    def test_manual_hijack_holds_until_released(self):
        self.assertEqual(self.gate.update(_result(0, manual=True)), (["10.0.0.9"], []))
        self.assertEqual(self.gate.update(_result(0, manual=True)), ([], []))
        self.assertEqual(self.gate.update(_result(0, manual=False)), ([], ["10.0.0.9"]))

    def test_clear_drops_every_hold(self):
        self.gate.update(_result(0, _action("level1", "e1", 1)))
        self.gate.clear()
        self.assertEqual(self.gate.update(_result(0, _action("recovery", "e1"))), ([], []))


class _FakeExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, ip):
        self.submitted.append(ip)

    def shutdown(self, wait=False):
        pass


class _FakeSignal:
    def __init__(self):
        self.requests = 0

    def request(self):
        self.requests += 1


def _runtime_config(tmp: Path, actions: dict) -> SimpleNamespace:
    path = tmp / "alarm.json"
    path.write_text(json.dumps({**_BASE_CONFIG, "actions": actions}), encoding="utf-8")
    zone = SimpleNamespace(speakers=["10.0.0.9"], smart_plugs=["10.0.0.31"])
    return SimpleNamespace(
        sid_to_zone={0: zone},
        speaker_auth=SimpleNamespace(username="u", password="p"),
        plug_auth=SimpleNamespace(email="e", password="p"),
        envs=SimpleNamespace(
            enable_speaker=True,
            enable_smart_plug=True,
            speaker_audio_file="clip.wav",
            alarm_config_path=str(path),
            audit_log_path="",
        ),
    )


class DeviceHandlerRoutingTest(unittest.TestCase):
    def _drive(self, handler, ip):
        executor = _FakeExecutor()
        signal = _FakeSignal()
        handler._executor = executor
        handler._cancel_signals = {ip: signal}

        handler.handle(_result(0, _action("level1", "e1", 1)), None)
        self.assertEqual(executor.submitted, [ip])
        self.assertEqual(handler.get_active_devices(), {ip})

        # Later levels of the same event do not restart an ACTIVE device.
        handler.handle(_result(0, _action("level2", "e1", 2)), None)
        self.assertEqual(executor.submitted, [ip])

        handler.handle(_result(0, _action("recovery", "e1")), None)
        self.assertEqual(signal.requests, 1)

    def test_speaker_on_at_level1_off_at_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = SpeakerHandler(_runtime_config(Path(tmpdir), {"speaker": {"levels": [1]}}), shm_config=None)
            self._drive(handler, "10.0.0.9")

    def test_smart_plug_on_at_level1_off_at_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = SmartPlugHandler(_runtime_config(Path(tmpdir), {"smart_plug": {"levels": [1]}}), shm_config=None)
            self._drive(handler, "10.0.0.31")

    def test_recovery_does_not_cancel_a_ready_device(self):
        """A pre-armed cancel on a READY device would kill its next activation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = SpeakerHandler(_runtime_config(Path(tmpdir), {"speaker": {"levels": [2]}}), shm_config=None)
            handler._executor = _FakeExecutor()
            signal = _FakeSignal()
            handler._cancel_signals = {"10.0.0.9": signal}
            handler.handle(_result(0, _action("level1", "e1", 1)), None)
            handler.handle(_result(0, _action("recovery", "e1")), None)
            self.assertEqual(handler._executor.submitted, [])
            self.assertEqual(signal.requests, 0)


if __name__ == "__main__":
    unittest.main()
