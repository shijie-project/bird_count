from __future__ import annotations

import contextlib
import csv
import json
import shutil
import struct
import sys
import zlib
from collections import defaultdict, deque
from pathlib import Path

from birdcount.deploy.alarm.config import AppConfig
from birdcount.deploy.alarm.models import AlarmAction, AlarmRules, CountSample
from birdcount.deploy.alarm.time_utils import new_event_id, safe_name, snapshot_name, utc_iso


# Columns of counts.csv and pre_window_counts.csv (the delivery package's layout).
COUNT_FIELDS = [
    "timestamp",
    "timestamp_iso",
    "camera_id",
    "count",
    "decision_count",
    "centroid_x",
    "centroid_y",
    "motion_excluded",
    "motion_reason",
    "motion_velocity_x_norm_per_sec",
    "motion_velocity_y_norm_per_sec",
    "motion_speed_norm_per_sec",
    "motion_net_displacement_norm",
    "motion_direction_degrees",
    "motion_direction_consistency",
]


# Windows' limit on a whole path unless long paths are enabled (MAX_PATH less the NUL).
WINDOWS_MAX_PATH = 259
# The longest action type and timestamp give the longest names.
_LONGEST_ACTION = "recovery"
_LONGEST_TIME = 4102444799.0  # 2099-12-31 23:59:59
_UUID_LENGTH = 36  # the dry-run worker's image id


def event_folder(camera_id: str, event_id: str) -> str:
    """An event's folder under events/<camera>/: its id without the camera, which the parent already names."""
    return safe_name(event_id).removeprefix(f"{safe_name(camera_id)}_")


def event_snapshot_name(action_type: str, timestamp: float, suffix: str) -> str:
    """File name of an action's frame inside its event folder."""
    return f"{safe_name(action_type)}_{int(timestamp)}{suffix}"


def longest_evidence_path(output_dir: Path, config: AppConfig) -> int:
    """Length of the longest file path the alarm evidence can reach for these cameras:
    a frame in snapshots/, its copy in events/<camera>/<event>/, or the dry-run
    worker's copy of that in mock_worker/images/<project>/."""
    root = Path(output_dir).resolve()
    copy = event_snapshot_name(_LONGEST_ACTION, _LONGEST_TIME, ".jpg")
    mock = root / "mock_worker" / "images" / safe_name(config.worker.project_slug) / f"{'x' * _UUID_LENGTH}-{copy}"
    worst = len(str(mock))
    for camera_id in config.cameras:
        event_id = new_event_id(camera_id, _LONGEST_TIME)
        frame = root / "snapshots" / snapshot_name(event_id, _LONGEST_ACTION, _LONGEST_TIME)
        in_event = root / "events" / safe_name(camera_id) / event_folder(camera_id, event_id) / copy
        worst = max(worst, len(str(frame)), len(str(in_event)))
    return worst


def long_paths_enabled() -> bool:
    """Whether Windows lets this process use paths past WINDOWS_MAX_PATH (always true elsewhere)."""
    if sys.platform != "win32":
        return True
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def path_length_problem(output_dir: Path, config: AppConfig) -> str | None:
    """Why evidence for these cameras could fail to save, or None.

    Evidence paths add ~90 characters to the output folder, so only a very deep
    folder breaks Windows' limit; the alarm frame would then fail to save at the
    moment of an alarm.
    """
    longest = longest_evidence_path(output_dir, config)
    if longest <= WINDOWS_MAX_PATH or long_paths_enabled():
        return None
    return (
        f"evidence paths under {output_dir} reach {longest} characters, past Windows' {WINDOWS_MAX_PATH}: "
        "alarm snapshots would fail to save. Enable long paths (LongPathsEnabled=1 under "
        r"HKLM\SYSTEM\CurrentControlSet\Control\FileSystem, as administrator) or move the output folder "
        "(ALARM_OUTPUT_DIR) higher up."
    )


def _fmt(value: float | None, spec: str) -> str:
    return "" if value is None else format(value, spec)


def count_row(sample: CountSample) -> dict[str, str | int]:
    """One sample as a counts.csv row, formatted exactly as the delivery package does."""
    return {
        "timestamp": f"{sample.timestamp:.3f}",
        "timestamp_iso": utc_iso(sample.timestamp),
        "camera_id": sample.camera_id,
        "count": f"{sample.count:.3f}",
        "decision_count": _fmt(sample.decision_count, ".3f"),
        "centroid_x": _fmt(sample.centroid_x, ".6f"),
        "centroid_y": _fmt(sample.centroid_y, ".6f"),
        "motion_excluded": int(sample.motion_excluded),
        "motion_reason": sample.motion_reason or "",
        "motion_velocity_x_norm_per_sec": _fmt(sample.motion_velocity_x_norm_per_sec, ".6f"),
        "motion_velocity_y_norm_per_sec": _fmt(sample.motion_velocity_y_norm_per_sec, ".6f"),
        "motion_speed_norm_per_sec": _fmt(sample.motion_speed_norm_per_sec, ".6f"),
        "motion_net_displacement_norm": _fmt(sample.motion_net_displacement_norm, ".6f"),
        "motion_direction_degrees": _fmt(sample.motion_direction_degrees, ".2f"),
        "motion_direction_consistency": _fmt(sample.motion_direction_consistency, ".6f"),
    }


class EvidenceStore:
    def __init__(self, output_dir: str | Path, rules: AlarmRules):
        self.output_dir = Path(output_dir)
        self.rules = rules
        self.events_dir = self.output_dir / "events"
        self.snapshots_dir = self.output_dir / "snapshots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._history: dict[str, deque[CountSample]] = defaultdict(deque)

    def observe(self, sample: CountSample) -> None:
        history = self._history[sample.camera_id]
        history.append(sample)
        min_ts = sample.timestamp - self.rules.evidence_window_seconds
        while history and history[0].timestamp < min_ts:
            history.popleft()

    def append_event_count(self, camera_id: str, event_id: str | None, sample: CountSample) -> None:
        if not event_id:
            return
        event_dir = self._event_dir(camera_id, event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        path = event_dir / "counts.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COUNT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(count_row(sample))

    def record_action(self, action: AlarmAction, sample: CountSample) -> dict:
        event_dir = self._event_dir(action.camera_id, action.event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._snapshot_for_action(event_dir, action, sample)
        self._write_pre_window(event_dir, action.camera_id)

        payload = {
            "action_type": action.action_type,
            "camera_id": action.camera_id,
            "event_id": action.event_id,
            "timestamp": action.timestamp,
            "timestamp_iso": utc_iso(action.timestamp),
            "count": action.count,
            "threshold": action.threshold,
            "level": action.level,
            "message": action.message,
            "snapshot_path": str(snapshot_path),
        }
        self._append_jsonl(self.output_dir / "events.jsonl", payload)
        self._append_jsonl(event_dir / "actions.jsonl", payload)
        if action.action_type == "recovery":
            self._write_event_summary(event_dir, payload)
        return payload

    def record_notification(self, action: AlarmAction, notify_result: dict) -> None:
        event_dir = self._event_dir(action.camera_id, action.event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "action_type": action.action_type,
            "camera_id": action.camera_id,
            "event_id": action.event_id,
            "timestamp": action.timestamp,
            "timestamp_iso": utc_iso(action.timestamp),
            "notification": notify_result,
        }
        self._append_jsonl(self.output_dir / "notifications.jsonl", payload)
        self._append_jsonl(event_dir / "notifications.jsonl", payload)

    def _event_dir(self, camera_id: str, event_id: str) -> Path:
        return self.events_dir / safe_name(camera_id) / event_folder(camera_id, event_id)

    def _snapshot_for_action(self, event_dir: Path, action: AlarmAction, sample: CountSample) -> Path:
        if sample.frame_path:
            src = Path(sample.frame_path)
            if src.exists():
                dst = event_dir / event_snapshot_name(action.action_type, action.timestamp, src.suffix)
                shutil.copy2(src, dst)
                return dst

        dst = event_dir / f"{safe_name(action.action_type)}_snapshot.png"
        color = {
            "level1": (245, 124, 0),
            "level2": (230, 81, 0),
            "level3": (211, 47, 47),
            "recovery": (35, 181, 91),
        }.get(action.action_type, (64, 112, 244))
        write_placeholder_png(dst, color=color, count=action.count, threshold=action.threshold)
        with dst.with_suffix(".txt").open("w", encoding="utf-8") as f:
            f.write(action.message + "\n")
            f.write(f"camera_id={action.camera_id}\n")
            f.write(f"event_id={action.event_id}\n")
            f.write(f"timestamp={utc_iso(action.timestamp)}\n")
        return dst

    def _write_pre_window(self, event_dir: Path, camera_id: str) -> None:
        with (event_dir / "pre_window_counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COUNT_FIELDS)
            writer.writeheader()
            writer.writerows(count_row(sample) for sample in self._history.get(camera_id, ()))

    def _write_event_summary(self, event_dir: Path, payload: dict) -> None:
        counts_path = event_dir / "counts.csv"
        counts = []
        if counts_path.exists():
            with counts_path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    with contextlib.suppress(KeyError, ValueError):
                        counts.append(float(row["count"]))
        summary = {
            "event_id": payload["event_id"],
            "camera_id": payload["camera_id"],
            "closed_at": payload["timestamp_iso"],
            "num_count_samples": len(counts),
            "max_count": max(counts) if counts else None,
            "mean_count": sum(counts) / len(counts) if counts else None,
            "recovery_action": payload,
        }
        with (event_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_placeholder_png(path: Path, color: tuple[int, int, int], count: float, threshold: float) -> None:
    """Write a simple valid PNG using only the standard library.

    The image is a colored evidence placeholder for dry-run/simulation mode.
    Real integration should pass the current camera frame via CountSample.frame_path.
    """
    width, height = 640, 360
    bg = (246, 248, 252)
    raw_rows = []
    ratio = max(0.0, min(1.0, count / max(threshold, 1.0)))
    bar_width = int((width - 80) * ratio)
    for y in range(height):
        row = bytearray()
        for x in range(width):
            pixel = bg
            if 30 <= y <= 90:
                pixel = color
            if 150 <= y <= 220 and 40 <= x <= 40 + bar_width:
                pixel = color
            if 150 <= y <= 220 and 40 + bar_width < x <= width - 40:
                pixel = (220, 226, 235)
            if x < 4 or y < 4 or x >= width - 4 or y >= height - 4:
                pixel = (36, 45, 62)
            row.extend(pixel)
        raw_rows.append(b"\x00" + bytes(row))
    _write_png_rgb(path, width, height, b"".join(raw_rows))


def _write_png_rgb(path: Path, width: int, height: int, raw_scanlines: bytes) -> None:
    def chunk(name: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(raw_scanlines, level=6))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
