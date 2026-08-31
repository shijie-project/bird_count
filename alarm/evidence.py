from __future__ import annotations

import csv
import json
import shutil
import struct
import zlib
from collections import defaultdict, deque
from pathlib import Path

from .models import AlarmAction, AlarmRules, CountSample
from .time_utils import safe_name, utc_iso


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
        fieldnames = [
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
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": f"{sample.timestamp:.3f}",
                    "timestamp_iso": utc_iso(sample.timestamp),
                    "camera_id": sample.camera_id,
                    "count": f"{sample.count:.3f}",
                    "decision_count": "" if sample.decision_count is None else f"{sample.decision_count:.3f}",
                    "centroid_x": "" if sample.centroid_x is None else f"{sample.centroid_x:.6f}",
                    "centroid_y": "" if sample.centroid_y is None else f"{sample.centroid_y:.6f}",
                    "motion_excluded": int(sample.motion_excluded),
                    "motion_reason": sample.motion_reason or "",
                    "motion_velocity_x_norm_per_sec": ""
                    if sample.motion_velocity_x_norm_per_sec is None
                    else f"{sample.motion_velocity_x_norm_per_sec:.6f}",
                    "motion_velocity_y_norm_per_sec": ""
                    if sample.motion_velocity_y_norm_per_sec is None
                    else f"{sample.motion_velocity_y_norm_per_sec:.6f}",
                    "motion_speed_norm_per_sec": ""
                    if sample.motion_speed_norm_per_sec is None
                    else f"{sample.motion_speed_norm_per_sec:.6f}",
                    "motion_net_displacement_norm": ""
                    if sample.motion_net_displacement_norm is None
                    else f"{sample.motion_net_displacement_norm:.6f}",
                    "motion_direction_degrees": ""
                    if sample.motion_direction_degrees is None
                    else f"{sample.motion_direction_degrees:.2f}",
                    "motion_direction_consistency": ""
                    if sample.motion_direction_consistency is None
                    else f"{sample.motion_direction_consistency:.6f}",
                }
            )

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
        return self.events_dir / safe_name(camera_id) / safe_name(event_id)

    def _snapshot_for_action(self, event_dir: Path, action: AlarmAction, sample: CountSample) -> Path:
        stage = safe_name(action.action_type)
        if sample.frame_path:
            src = Path(sample.frame_path)
            if src.exists():
                dst = event_dir / f"{stage}_{src.name}"
                shutil.copy2(src, dst)
                return dst

        dst = event_dir / f"{stage}_snapshot.png"
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
        rows = list(self._history.get(camera_id, ()))
        path = event_dir / "pre_window_counts.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
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
                ],
            )
            writer.writeheader()
            for sample in rows:
                writer.writerow(
                    {
                        "timestamp": f"{sample.timestamp:.3f}",
                        "timestamp_iso": utc_iso(sample.timestamp),
                        "camera_id": sample.camera_id,
                        "count": f"{sample.count:.3f}",
                        "decision_count": "" if sample.decision_count is None else f"{sample.decision_count:.3f}",
                        "centroid_x": "" if sample.centroid_x is None else f"{sample.centroid_x:.6f}",
                        "centroid_y": "" if sample.centroid_y is None else f"{sample.centroid_y:.6f}",
                        "motion_excluded": int(sample.motion_excluded),
                        "motion_reason": sample.motion_reason or "",
                        "motion_velocity_x_norm_per_sec": ""
                        if sample.motion_velocity_x_norm_per_sec is None
                        else f"{sample.motion_velocity_x_norm_per_sec:.6f}",
                        "motion_velocity_y_norm_per_sec": ""
                        if sample.motion_velocity_y_norm_per_sec is None
                        else f"{sample.motion_velocity_y_norm_per_sec:.6f}",
                        "motion_speed_norm_per_sec": ""
                        if sample.motion_speed_norm_per_sec is None
                        else f"{sample.motion_speed_norm_per_sec:.6f}",
                        "motion_net_displacement_norm": ""
                        if sample.motion_net_displacement_norm is None
                        else f"{sample.motion_net_displacement_norm:.6f}",
                        "motion_direction_degrees": ""
                        if sample.motion_direction_degrees is None
                        else f"{sample.motion_direction_degrees:.2f}",
                        "motion_direction_consistency": ""
                        if sample.motion_direction_consistency is None
                        else f"{sample.motion_direction_consistency:.6f}",
                    }
                )

    def _write_event_summary(self, event_dir: Path, payload: dict) -> None:
        counts_path = event_dir / "counts.csv"
        counts = []
        if counts_path.exists():
            with counts_path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        counts.append(float(row["count"]))
                    except (KeyError, ValueError):
                        pass
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
