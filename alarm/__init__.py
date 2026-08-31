"""Chicken pile-up alarm core — pure stdlib, no torch / cv2 / numpy.

Vendored from the `chicken_alarm_delivery_20260814` package (`chicken_alarm/`).
Kept dependency-free on purpose so it can run inside the realtime consumer
thread without dragging the delivery package's model runtime along.

Deltas from the upstream copy, all additive or subtractive — no logic changes:

* `OriginalRuntimeConfig` / `original_runtime.stream_id_to_camera_id` dropped.
  Stream -> camera mapping now comes from `topology.yaml` via
  `runtime.config.Config.sid_to_mac`; see `camera_ids.py`.
* `WorkerNotifier(mock_root=...)` is now required rather than defaulting to a
  path inside the delivery package, so dry-run artifacts land under this
  project's output dir.
* `AlarmManager.reset_all()` / `.active_cameras()` and `CameraAlarmState.reset()`
  added for the GUI cancel-all + active-device surface.
* `camera_ids.py` is new (bare MAC -> `axisN/MAC` resolution).
* Reformatted with this project's ruff profile, and one dead `typing.Iterable`
  import dropped from `evidence.py`.

Everything else — the Level 1/2/3 + recovery state machine, the directional
motion filter, the evidence layout and the SMS worker payload — is byte-identical
to the delivery package, so its reports and offline tooling still read our output.

The realtime wiring lives in `runtime.handlers.sms_alarm`.
"""

from .config import AppConfig, load_config
from .evidence import EvidenceStore
from .models import AlarmAction, AlarmRules, CameraConfig, CountSample, MotionFilterConfig, WorkerConfig
from .motion_filter import DirectionalMotionFilter, MotionDecision
from .sms_worker import WorkerNotifier
from .state_machine import AlarmManager


__all__ = [
    "AlarmAction",
    "AlarmManager",
    "AlarmRules",
    "AppConfig",
    "CameraConfig",
    "CountSample",
    "DirectionalMotionFilter",
    "EvidenceStore",
    "MotionDecision",
    "MotionFilterConfig",
    "WorkerConfig",
    "WorkerNotifier",
    "load_config",
]
