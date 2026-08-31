from __future__ import annotations

import uuid
from dataclasses import dataclass

from .models import AlarmAction, AlarmRules, CameraConfig, CountSample
from .time_utils import safe_name, utc_iso


@dataclass
class CameraRuntimeState:
    status: str = "normal"
    breach_since: float | None = None
    clear_since: float | None = None
    event_id: str | None = None
    event_opened_at: float | None = None
    level: int = 0
    last_alert_at: float | None = None


class CameraAlarmState:
    """State machine for one camera.

    States:
    - normal: no active event
    - active: event is open and count is currently over threshold
    - recovering: event is open and count is currently below threshold
    """

    def __init__(self, camera: CameraConfig, rules: AlarmRules):
        self.camera = camera
        self.rules = rules
        self.state = CameraRuntimeState()

    @property
    def active_event_id(self) -> str | None:
        return self.state.event_id

    def process(self, sample: CountSample) -> list[AlarmAction]:
        if sample.camera_id != self.camera.camera_id:
            raise ValueError(f"sample camera {sample.camera_id!r} does not match {self.camera.camera_id!r}")

        actions: list[AlarmAction] = []
        decision_count = sample.decision_count if sample.decision_count is not None else sample.count
        over_threshold = decision_count >= self.camera.threshold

        if self.state.status == "normal":
            if over_threshold:
                if self.state.breach_since is None:
                    self.state.breach_since = sample.timestamp
                if sample.timestamp - self.state.breach_since >= self.rules.alert_trigger_seconds:
                    self._open_event(sample.timestamp)
                    actions.append(self._alert_action(sample, level=1))
            else:
                self.state.breach_since = None
            return actions

        if self.state.status == "active":
            if over_threshold:
                actions.extend(self._maybe_escalate(sample))
            else:
                self.state.status = "recovering"
                self.state.clear_since = sample.timestamp
            return actions

        if self.state.status == "recovering":
            if over_threshold:
                self.state.status = "active"
                self.state.clear_since = None
                actions.extend(self._maybe_escalate(sample))
            else:
                if self.state.clear_since is None:
                    self.state.clear_since = sample.timestamp
                if sample.timestamp - self.state.clear_since >= self.rules.recovery_seconds:
                    actions.append(self._recovery_action(sample))
                    self._reset()
            return actions

        raise RuntimeError(f"unknown camera alarm state: {self.state.status}")

    def _open_event(self, timestamp: float) -> None:
        stamp = utc_iso(timestamp).replace(":", "").replace("-", "")
        self.state.status = "active"
        self.state.event_id = f"{safe_name(self.camera.camera_id)}_{stamp}_{uuid.uuid4().hex[:8]}"
        self.state.event_opened_at = timestamp
        self.state.level = 0
        self.state.last_alert_at = None
        self.state.clear_since = None

    def _maybe_escalate(self, sample: CountSample) -> list[AlarmAction]:
        if self.state.level >= self.rules.max_level:
            return []
        if self.state.last_alert_at is None:
            return []
        if sample.timestamp - self.state.last_alert_at < self.rules.escalation_seconds:
            return []
        return [self._alert_action(sample, level=self.state.level + 1)]

    def _alert_action(self, sample: CountSample, level: int) -> AlarmAction:
        self.state.level = level
        self.state.last_alert_at = sample.timestamp
        event_id = self.state.event_id
        if event_id is None:
            raise RuntimeError("cannot create alert action without an active event")
        return AlarmAction(
            action_type=f"level{level}",
            camera_id=self.camera.camera_id,
            event_id=event_id,
            timestamp=sample.timestamp,
            count=sample.count,
            threshold=self.camera.threshold,
            level=level,
            message=self._alert_message(sample, level),
        )

    def _recovery_action(self, sample: CountSample) -> AlarmAction:
        event_id = self.state.event_id
        if event_id is None:
            raise RuntimeError("cannot recover without an active event")
        if sample.motion_excluded:
            message = (
                "Movement Exclusion Notice: Dense chickens are moving in one direction, "
                "so this is not treated as a pile-up.\n"
                f"Camera ID: {self.camera.camera_id}\n"
                "Screenshot: {image_url}\n"
                f"Count: N={sample.count:.1f}, Threshold T={self.camera.threshold:.1f}\n"
                "Status: No pile-up alert is active for this movement."
            )
        else:
            message = (
                "Recovery Notice: Chicken pile-up condition has cleared.\n"
                f"Camera ID: {self.camera.camera_id}\n"
                "Screenshot: {image_url}\n"
                f"Count: N={sample.count:.1f}, Threshold T={self.camera.threshold:.1f}\n"
                "Status: The event has been reset."
            )
        return AlarmAction(
            action_type="recovery",
            camera_id=self.camera.camera_id,
            event_id=event_id,
            timestamp=sample.timestamp,
            count=sample.count,
            threshold=self.camera.threshold,
            level=None,
            message=message,
        )

    def _alert_message(self, sample: CountSample, level: int) -> str:
        if level == 1:
            header = "Level 1 Alert: Chicken pile-up detected."
            action = "Action: Please check this camera."
        elif level == 2:
            header = "Level 2 Alert: Chicken pile-up is still detected after 3 minutes."
            action = "Action: Please inspect the area as soon as possible."
        else:
            header = "Level 3 Alert: Chicken pile-up is still detected after repeated checks."
            action = "Action: Urgent attention required. Repeated SMS alerts will pause until recovery."

        return (
            f"{header}\n"
            f"Camera ID: {self.camera.camera_id}\n"
            "Screenshot: {image_url}\n"
            f"Count: N={sample.count:.1f}, Threshold T={self.camera.threshold:.1f}\n"
            f"{action}"
        )

    def reset(self) -> None:
        """Drop any in-flight event and return to `normal` (operator cancel-all)."""
        self._reset()

    def _reset(self) -> None:
        self.state = CameraRuntimeState()


class AlarmManager:
    def __init__(self, cameras: dict[str, CameraConfig], rules: AlarmRules):
        self._states = {camera_id: CameraAlarmState(camera, rules) for camera_id, camera in cameras.items()}

    def process(self, sample: CountSample) -> list[AlarmAction]:
        state = self._states.get(sample.camera_id)
        if state is None:
            raise KeyError(f"unknown camera_id: {sample.camera_id}")
        return state.process(sample)

    def active_event_id(self, camera_id: str) -> str | None:
        state = self._states.get(camera_id)
        return state.active_event_id if state else None

    def active_cameras(self) -> set[str]:
        """Camera ids with an open (non-`normal`) event."""
        return {cid for cid, st in self._states.items() if st.state.status != "normal"}

    def reset_all(self) -> None:
        """Drop every in-flight event (operator cancel-all)."""
        for state in self._states.values():
            state.reset()
