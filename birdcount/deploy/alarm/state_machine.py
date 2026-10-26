from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

from birdcount.deploy.alarm.models import AlarmAction, AlarmRules, CameraConfig, CountSample
from birdcount.deploy.alarm.time_utils import new_event_id


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
    - active: event is open and the count still holds it (>= the clear level)
    - recovering: event is open and the count is below the clear level

    Decisions use the median count over `smoothing_seconds`. An event opens
    at the threshold and clears below `recovery_ratio` x threshold;
    escalating needs the count at the threshold itself.
    """

    def __init__(self, camera: CameraConfig, rules: AlarmRules):
        self.camera = camera
        self.rules = rules
        self.clear_level = camera.threshold * rules.recovery_ratio
        self.state = CameraRuntimeState()
        self._recent: deque[tuple[float, float]] = deque()

    @property
    def active_event_id(self) -> str | None:
        return self.state.event_id

    def process(self, sample: CountSample) -> list[AlarmAction]:
        if sample.camera_id != self.camera.camera_id:
            raise ValueError(f"sample camera {sample.camera_id!r} does not match {self.camera.camera_id!r}")

        actions: list[AlarmAction] = []
        value = self._decision_value(sample)
        over_threshold = value >= self.camera.threshold
        holds = value >= self.clear_level

        if self.state.status == "normal":
            if over_threshold:
                if self.state.breach_since is None:
                    self.state.breach_since = sample.timestamp
                if sample.timestamp - self.state.breach_since >= self.rules.alert_trigger_seconds:
                    self._open_event(sample.timestamp)
                    actions.append(self._alert_action(sample, value, level=1))
            else:
                self.state.breach_since = None
            return actions

        if self.state.status == "active":
            if not holds:
                self.state.status = "recovering"
                self.state.clear_since = sample.timestamp
            elif over_threshold:
                actions.extend(self._maybe_escalate(sample, value))
            return actions

        if self.state.status == "recovering":
            if holds:
                self.state.status = "active"
                self.state.clear_since = None
                if over_threshold:
                    actions.extend(self._maybe_escalate(sample, value))
            else:
                if self.state.clear_since is None:
                    self.state.clear_since = sample.timestamp
                if sample.timestamp - self.state.clear_since >= self.rules.recovery_seconds:
                    actions.append(self._recovery_action(sample, value))
                    self._reset()
            return actions

        raise RuntimeError(f"unknown camera alarm state: {self.state.status}")

    def _open_event(self, timestamp: float) -> None:
        self.state.status = "active"
        self.state.event_id = new_event_id(self.camera.camera_id, timestamp)
        self.state.event_opened_at = timestamp
        self.state.level = 0
        self.state.last_alert_at = None
        self.state.clear_since = None

    def _decision_value(self, sample: CountSample) -> float:
        """The count this sample decides on: the recent median, where a sample the
        motion filter excludes (a flock walking past) never holds an event open."""
        value = sample.decision_count if sample.decision_count is not None else sample.count
        if sample.motion_excluded:
            value = min(value, self.clear_level - 1e-3)
        if self.rules.smoothing_seconds <= 0:
            return value
        recent = self._recent
        recent.append((sample.timestamp, value))
        while recent[0][0] <= sample.timestamp - self.rules.smoothing_seconds:
            recent.popleft()
        return median(v for _, v in recent)

    def _maybe_escalate(self, sample: CountSample, value: float) -> list[AlarmAction]:
        if self.state.level >= self.rules.max_level:
            return []
        if self.state.last_alert_at is None:
            return []
        if sample.timestamp - self.state.last_alert_at < self.rules.escalation_seconds:
            return []
        return [self._alert_action(sample, value, level=self.state.level + 1)]

    def _alert_action(self, sample: CountSample, value: float, level: int) -> AlarmAction:
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
            count=value,
            threshold=self.camera.threshold,
            level=level,
            message=self._alert_message(value, level),
        )

    def _recovery_action(self, sample: CountSample, value: float) -> AlarmAction:
        event_id = self.state.event_id
        if event_id is None:
            raise RuntimeError("cannot recover without an active event")
        if sample.motion_excluded:
            message = (
                "Movement Exclusion Notice: Dense chickens are moving in one direction, "
                "so this is not treated as a pile-up.\n"
                f"Camera ID: {self.camera.camera_id}\n"
                "Screenshot: {image_url}\n"
                f"Count: N={value:.1f}, Threshold T={self.camera.threshold:.1f}\n"
                "Status: No pile-up alert is active for this movement."
            )
        else:
            message = (
                "Recovery Notice: Chicken pile-up condition has cleared.\n"
                f"Camera ID: {self.camera.camera_id}\n"
                "Screenshot: {image_url}\n"
                f"Count: N={value:.1f}, Threshold T={self.camera.threshold:.1f}\n"
                "Status: The event has been reset."
            )
        return AlarmAction(
            action_type="recovery",
            camera_id=self.camera.camera_id,
            event_id=event_id,
            timestamp=sample.timestamp,
            count=value,
            threshold=self.camera.threshold,
            level=None,
            message=message,
        )

    def _alert_message(self, value: float, level: int) -> str:
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
            f"Count: N={value:.1f}, Threshold T={self.camera.threshold:.1f}\n"
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

    def state_of(self, camera_id: str) -> CameraRuntimeState:
        """Current runtime state (status, level, event id) of one camera."""
        return self._states[camera_id].state

    def active_cameras(self) -> set[str]:
        """Camera ids with an open (non-`normal`) event."""
        return {cid for cid, st in self._states.items() if st.state.status != "normal"}

    def reset_all(self) -> None:
        """Drop every in-flight event (operator cancel-all)."""
        for state in self._states.values():
            state.reset()
