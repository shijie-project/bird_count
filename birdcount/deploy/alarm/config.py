from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from birdcount.deploy.alarm.models import AlarmRules, CameraConfig, MotionFilterConfig, WorkerConfig


# Which alarm levels each action handler reacts to when `actions` is missing
# from the config (or names no entry for that handler). Recovery is not a level
# and is never subscribed to: every handler hears it, so a device switched on at
# some level is always switched off again when the event closes.
DEFAULT_ACTION_LEVELS: dict[str, frozenset[int]] = {
    "sms": frozenset({1, 2, 3}),
    "speaker": frozenset({1}),
    "smart_plug": frozenset({1}),
}


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    sample_fps: float
    rules: AlarmRules
    motion_filter: MotionFilterConfig
    worker: WorkerConfig
    cameras: dict[str, CameraConfig]
    action_levels: dict[str, frozenset[int]]

    def levels_for(self, handler: str) -> frozenset[int]:
        """Alarm levels `handler` acts on (`sms`, `speaker`, `smart_plug`)."""
        return self.action_levels.get(handler, DEFAULT_ACTION_LEVELS.get(handler, frozenset()))

    def threshold_of(self, mac: str) -> float | None:
        """A camera's threshold by bare MAC; None when the config does not cover it."""
        from birdcount.deploy.alarm.camera_ids import mac_of

        return next((cam.threshold for cid, cam in self.cameras.items() if mac_of(cid) == mac), None)

    def with_thresholds(self, thresholds: dict[str, float]) -> AppConfig:
        """This config with the thresholds of some cameras (by MAC) replaced: trying others on recordings."""
        from birdcount.deploy.alarm.camera_ids import mac_of

        cameras = {
            cid: replace(cam, threshold=float(thresholds[mac_of(cid)])) if mac_of(cid) in thresholds else cam
            for cid, cam in self.cameras.items()
        }
        return replace(self, cameras=cameras)


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
    parts = value.split(",") if isinstance(value, str) else list(value)
    return tuple(str(part).strip() for part in parts if str(part).strip())


def _action_levels(raw: dict, max_level: int) -> dict[str, frozenset[int]]:
    """Parse `actions: {"<handler>": {"levels": [1, 2]}}` on top of the defaults."""
    levels = dict(DEFAULT_ACTION_LEVELS)
    for handler, entry in (raw or {}).items():
        values = entry.get("levels", []) if isinstance(entry, dict) else entry
        parsed = frozenset(int(v) for v in values)
        bad = sorted(v for v in parsed if not 1 <= v <= max_level)
        if bad:
            raise ValueError(f"actions.{handler}.levels has {bad}; levels run 1..{max_level}")
        levels[str(handler)] = parsed
    return levels


def _merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; lists and scalars in `override` replace wholesale."""
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Read a config, applying `"extends": "<other.json>"` (relative to this file).

    Lets a variant — e.g. configs/alarm_debug.json with shortened timings —
    override a few keys while the 20 per-camera thresholds stay in one place.
    """
    if path.resolve() in seen:
        raise ValueError(f"alarm config `extends` loop through {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _merge(_load_raw(path.parent / parent, (*seen, path.resolve())), raw)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    raw = _load_raw(path)

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

    max_level = _int(rules_raw, "max_level", 3)
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
            max_level=max_level,
            recovery_ratio=_float(rules_raw, "recovery_ratio", AlarmRules.recovery_ratio),
            smoothing_seconds=_float(rules_raw, "smoothing_seconds", AlarmRules.smoothing_seconds),
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
            timeout_seconds=_float(worker_raw, "timeout_seconds", 20.0),
            mock_recipients=_string_tuple(worker_raw, "mock_recipients", ("+61400000000",)),
        ),
        cameras=cameras,
        action_levels=_action_levels(raw.get("actions", {}), max_level),
    )
