"""Settings read by the live runtime."""

from __future__ import annotations

from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator

from birdcount.core.settings import CoreSettings, OptionalProjectPath, ProjectPath


def parse_hours(spec: str) -> tuple[dtime, dtime]:
    """'06:00-20:00' -> (06:00, 20:00); the end is exclusive."""
    try:
        start, end = (datetime.strptime(part.strip(), "%H:%M").time() for part in spec.split("-"))
    except ValueError as e:
        raise ValueError(f"expected 'HH:MM-HH:MM', got {spec!r}") from e
    if start >= end:
        raise ValueError(f"start must be before end, got {spec!r}")
    return start, end


class DeploySettings(CoreSettings):
    # Debug GUI panel; with SMS in dry-run, its manual trigger also simulates a
    # pile-up through the whole alarm instead of only switching devices.
    debug: bool = False

    # --- Cameras -------------------------------------------------------------------
    # Zones: their cameras (IP + MAC), speakers and smart plugs.
    topology_path: ProjectPath = Path("configs/topology.yaml")
    # camera: every camera in the topology, live. video: the clips in
    # VIDEO_PATHS replayed as cameras (each takes the zone of the camera whose
    # MAC is in its filename).
    source_type: Literal["camera", "video"] = "camera"
    # Comma-separated video files and/or folders (every clip inside).
    video_paths: str = "../data/videos"
    # Use only the first N streams; blank = all.
    num_streams: int | None = Field(None, ge=1)
    # Stream URL per camera; {ip} is filled in. Credentials are added from
    # CAMERA_USERNAME / CAMERA_PASSWORD and never logged.
    camera_url: str = "https://{ip}/mjpg/1/video.mjpg"
    camera_username: str = "root"
    camera_password: SecretStr = SecretStr("root")

    # --- Data path -----------------------------------------------------------------
    # Inference rate per camera. Frames are decoded at the camera's own rate,
    # but only this many per second are handed to the model.
    fps: int = Field(10, gt=0)
    # Frame slots per camera in the shared-memory ring.
    num_buffers: int = Field(4, ge=3)
    # Inference processes; camera i goes to worker i % num_workers_per_gpu.
    num_workers_per_gpu: int = Field(1, ge=1)
    cuda_device: str = "0"
    # Decode video on the GPU's video engine (NVDEC via D3D11 on Windows),
    # falling back to software per stream when unavailable.
    hw_decode: bool = True

    # --- Alarm ---------------------------------------------------------------------
    # Thresholds, timing rules and which levels each action reacts to.
    alarm_config_path: ProjectPath = Path("configs/alarm.json")
    # Evidence and dry-run SMS output; blank = the `output_dir` in the config.
    alarm_output_dir: OptionalProjectPath = None
    enable_sms_alarm: bool = False
    # Off (default): SMS go through the dry-run path, payloads and snapshots are
    # written to disk and nothing is sent. On + FARM_SMS_API_KEY: real SMS.
    sms_alarm_real_worker: bool = False
    farm_sms_api_key: SecretStr = SecretStr("")

    enable_speaker: bool = False
    speaker_username: str = "admin"
    speaker_password: SecretStr = SecretStr("admin")
    # The deterrent clip; must already be uploaded to every speaker under this name.
    speaker_audio_file: str = "7MB.wav"

    enable_smart_plug: bool = False
    tapo_email: str = ""
    tapo_password: SecretStr = SecretStr("")

    # --- Monitor -------------------------------------------------------------------
    # The camera grid window; also switchable from the operator panel.
    enable_monitor: bool = False
    # Refresh rate of the grid. The grabbers retrieve frames at this rate too
    # while the monitor is open, so the picture moves smoothly between inferences.
    monitor_fps: float = Field(10.0, gt=0)
    # Overlays at start; both switchable from the debug panel.
    show_density_map: bool = False
    show_mask: bool = False

    # --- Records -------------------------------------------------------------------
    # One CSV row per camera per second plus review snapshots, inside the local
    # time window, for re-tuning thresholds on real farm data.
    enable_count_logger: bool = False
    count_log_dir: ProjectPath = Path("outputs/count_log")
    count_log_hours: str = "06:00-20:00"
    # Continuous recording of every camera into fixed-length segments, at the
    # inference rate; also switchable per camera from the debug panel.
    enable_video_recorder: bool = False
    video_record_dir: ProjectPath = Path("recordings")
    video_segment_seconds: float = Field(300.0, gt=0)
    # mp4v plays everywhere; a fourcc this machine cannot encode falls back to it.
    video_fourcc: str = Field("mp4v", min_length=4, max_length=4)
    # Operator and alarm actions as JSONL; blank disables.
    audit_log_path: OptionalProjectPath = Path("logs/audit.jsonl")

    @field_validator("num_streams", mode="before")
    @classmethod
    def _blank_is_all(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("camera_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if "{ip}" not in value:
            raise ValueError(f"must contain {{ip}}, got {value!r}")
        return value

    @field_validator("count_log_hours")
    @classmethod
    def _check_hours(cls, value: str) -> str:
        parse_hours(value)
        return value
