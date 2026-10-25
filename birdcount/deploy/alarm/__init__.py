"""Pile-up alarm core: Level 1/2/3 + recovery state machine, directional
motion filter, evidence store, SMS worker. Pure stdlib — no numpy, torch, cv2.

From the chicken_alarm_delivery_20260814 package. Its decision logic is
unchanged, and so is everything it writes — the evidence layout, CSV / JSON
contents and the SMS payload — so the delivery package's reports and offline
tooling read our output. Internals were tidied (one CSV row formatter instead
of two copies, the SMS API key passed in from the deploy settings instead of
read from the environment) and the operator surface added: `AlarmManager`'s
`reset_all` / `active_cameras` / `state_of`, `AppConfig.action_levels`, and
`camera_ids` (bare MAC -> the config's `axisN/MAC` ids).
"""

from birdcount.deploy.alarm.config import DEFAULT_ACTION_LEVELS, AppConfig, load_config
from birdcount.deploy.alarm.evidence import EvidenceStore
from birdcount.deploy.alarm.models import (
    AlarmAction,
    AlarmRules,
    CameraConfig,
    CountSample,
    MotionFilterConfig,
    WorkerConfig,
)
from birdcount.deploy.alarm.motion_filter import DirectionalMotionFilter, MotionDecision
from birdcount.deploy.alarm.sms_worker import WorkerNotifier
from birdcount.deploy.alarm.state_machine import AlarmManager


__all__ = [
    "DEFAULT_ACTION_LEVELS",
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
