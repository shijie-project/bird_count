"""Settings read by the live runtime."""

from __future__ import annotations

from datetime import datetime
from datetime import time as dtime
from pathlib import Path

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

    # --- Records -------------------------------------------------------------------
    # One CSV row per camera per second plus review snapshots, inside the local
    # time window, for re-tuning thresholds on real farm data.
    enable_count_logger: bool = False
    count_log_dir: ProjectPath = Path("outputs/count_log")
    count_log_hours: str = "06:00-20:00"
    # Operator and alarm actions as JSONL; blank disables.
    audit_log_path: OptionalProjectPath = Path("logs/audit.jsonl")

    @field_validator("count_log_hours")
    @classmethod
    def _check_hours(cls, value: str) -> str:
        parse_hours(value)
        return value
