from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import AlarmRules, CameraConfig, MotionFilterConfig, WorkerConfig


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    sample_fps: float
    rules: AlarmRules
    motion_filter: MotionFilterConfig
    worker: WorkerConfig
    cameras: dict[str, CameraConfig]


def _float(data: dict, key: str, default: float) -> float:
    value = data.get(key, default)
    return float(value)


def _int(data: dict, key: str, default: int) -> int:
    value = data.get(key, default)
    return int(value)


def _string_tuple(data: dict, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    return tuple(str(part).strip() for part in parts if str(part).strip())


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rules_raw = raw.get("rules", {})
    motion_raw = raw.get("motion_filter", {})
    worker_raw = raw.get("worker", {})
    cameras: dict[str, CameraConfig] = {}
    for item in raw.get("cameras", []):
        camera = CameraConfig(
            camera_id=str(item["camera_id"]),
            name=str(item.get("name") or item["camera_id"]),
            threshold=float(item["threshold"]),
        )
        cameras[camera.camera_id] = camera

    if not cameras:
        raise ValueError("config must contain at least one camera")

    return AppConfig(
        output_dir=(path.parent / raw.get("output_dir", "runs")).resolve()
        if not Path(raw.get("output_dir", "runs")).is_absolute()
        else Path(raw.get("output_dir", "runs")).resolve(),
        sample_fps=_float(raw, "sample_fps", 5.0),
        rules=AlarmRules(
            alert_trigger_seconds=_float(rules_raw, "alert_trigger_seconds", 10.0),
            escalation_seconds=_float(rules_raw, "escalation_seconds", 180.0),
            recovery_seconds=_float(rules_raw, "recovery_seconds", 30.0),
            evidence_window_seconds=_float(rules_raw, "evidence_window_seconds", 30.0),
            max_level=_int(rules_raw, "max_level", 3),
        ),
        motion_filter=MotionFilterConfig(
            enabled=bool(motion_raw.get("enabled", True)),
            window_seconds=_float(motion_raw, "window_seconds", 3.0),
            min_samples=_int(motion_raw, "min_samples", 6),
            smoothing_samples=_int(motion_raw, "smoothing_samples", 5),
            min_speed_norm_per_sec=_float(motion_raw, "min_speed_norm_per_sec", 0.018),
            min_net_displacement_norm=_float(motion_raw, "min_net_displacement_norm", 0.055),
            min_direction_consistency=_float(motion_raw, "min_direction_consistency", 0.8),
        ),
        worker=WorkerConfig(
            enabled=bool(worker_raw.get("enabled", True)),
            dry_run=bool(worker_raw.get("dry_run", True)),
            base_url=str(worker_raw.get("base_url", "https://r2api.specmora.com")).rstrip("/"),
            project_slug=str(worker_raw.get("project_slug", "default")),
            api_key_env=str(worker_raw.get("api_key_env", "FARM_SMS_API_KEY")),
            timeout_seconds=_float(worker_raw, "timeout_seconds", 20.0),
            mock_recipients=_string_tuple(worker_raw, "mock_recipients", ("+61400000000",)),
        ),
        cameras=cameras,
    )
