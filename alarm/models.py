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
    api_key_env: str = "FARM_SMS_API_KEY"
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
