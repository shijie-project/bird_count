from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    threshold: float


@dataclass(frozen=True)
class AlarmRules:
    alert_trigger_seconds: float = 10.0
    escalation_seconds: float = 180.0
    recovery_seconds: float = 30.0
    evidence_window_seconds: float = 30.0
    max_level: int = 3
    # Hysteresis: an open event clears only once the count is below this
    # fraction of the threshold, so a count hovering at it does not open and
    # close an event (and text the farm) every minute. 1.0 = no hysteresis.
    recovery_ratio: float = 0.9
    # Decisions use the median count over this many seconds, so one noisy
    # frame neither breaks a 10 s breach nor a 30 s recovery. 0 = per frame.
    smoothing_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.recovery_ratio <= 1:
            raise ValueError(f"recovery_ratio must be in (0, 1], got {self.recovery_ratio}")
        if self.smoothing_seconds < 0:
            raise ValueError(f"smoothing_seconds must be >= 0, got {self.smoothing_seconds}")


@dataclass(frozen=True)
class MotionFilterConfig:
    enabled: bool = True
    window_seconds: float = 3.0
    min_samples: int = 6
    smoothing_samples: int = 5
    min_speed_norm_per_sec: float = 0.018
    min_net_displacement_norm: float = 0.055
    min_direction_consistency: float = 0.8


@dataclass(frozen=True)
class WorkerConfig:
    enabled: bool = True
    dry_run: bool = True
    base_url: str = "https://r2api.specmora.com"
    project_slug: str = "default"
    timeout_seconds: float = 20.0
    mock_recipients: tuple[str, ...] = ("+61400000000",)


@dataclass(frozen=True)
class CountSample:
    timestamp: float
    camera_id: str
    count: float
    frame_path: str | None = None
    centroid_x: float | None = None
    centroid_y: float | None = None
    decision_count: float | None = None
    motion_excluded: bool = False
    motion_reason: str | None = None
    motion_velocity_x_norm_per_sec: float | None = None
    motion_velocity_y_norm_per_sec: float | None = None
    motion_speed_norm_per_sec: float | None = None
    motion_net_displacement_norm: float | None = None
    motion_direction_degrees: float | None = None
    motion_direction_consistency: float | None = None


@dataclass(frozen=True)
class AlarmAction:
    action_type: str
    camera_id: str
    event_id: str
    timestamp: float
    count: float
    threshold: float
    level: int | None
    message: str
