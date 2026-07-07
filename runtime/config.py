"""Configuration root.

`EnvSettings` reads from `.env` / system env vars via pydantic-settings.
`Config` wraps that input plus the YAML topology and runtime-derived fields
(device, stream sources, sid mappings) into a single source of truth that the
rest of `runtime/` reads from.
"""

import logging
import sys
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


# ======================================================================
# Exceptions
# ======================================================================


class ConfigError(Exception):
    """Raised on invalid or incomplete system configuration."""


# ======================================================================
# Sub-configs (declarative; aggregated by `Config` below)
# ======================================================================


class SHMSettings(BaseModel):
    """RAM-sizing inputs for the SHM ring buffer.

    Distinct from `runtime.shared_memory.SharedMemoryConfig`, which is the
    *runtime descriptor* (block shapes, dtypes) — built from these settings
    by `TaskDispatcher`.
    """

    name_prefix: str = "vstream_shm"

    # Double buffering is the floor; 4–6 buffers is the sweet spot for ~20
    # streams with variable network latency.
    num_buffers: int = Field(default=4, ge=2)

    # Storage resolution held in RAM. Usually matches the camera source.
    height: int = 720
    width: int = 1080
    channels: int = 3

    @property
    def frame_bytes(self) -> int:
        return self.height * self.width * self.channels

    @property
    def block_size(self) -> int:
        """Total bytes for ONE stream's buffer ring."""
        return self.frame_bytes * self.num_buffers


class ModelConfig(BaseModel):
    """AI inference engine configuration."""

    path: Path

    @field_validator("path")
    @classmethod
    def _check_exists(cls, v: Path) -> Path:
        # Warn-not-fail: container builds may mount the weights later.
        if not v.exists():
            logger.warning("Model file not found at: %s. Ensure it is mounted correctly.", v)
        return v


class ZoneConfig(BaseModel):
    """A physical zone with cameras + IoT devices addressed by IP."""

    name: str
    cameras: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    smart_plugs: list[str] = Field(default_factory=list)

    thresholds: Union[int, list[int]] = 60

    @field_validator("cameras", mode="before")
    @classmethod
    def _normalize_cameras(cls, v):
        if v is None:
            return []
        out: list[str] = []
        for entry in v:
            out.append(str(entry))
        return out

    @model_validator(mode="after")
    def _broadcast_thresholds(self) -> "ZoneConfig":
        """Lift a scalar threshold to one-per-camera so downstream code iterates uniformly."""
        if isinstance(self.thresholds, int):
            self.thresholds = [self.thresholds] * len(self.cameras)
        return self


class SmartPlugAuthConfig(BaseModel):
    """Credentials for external device control. Sourced from `EnvSettings.tapo_*`."""

    email: str = ""
    password: str = ""


# ======================================================================
# Root input: environment-driven settings
# ======================================================================


class EnvSettings(BaseSettings):
    """Environment-level settings loaded from .env / system env vars."""

    # --- Runtime / debugging ---
    debug: bool = False
    verbose_debug: bool = False

    # --- Model & input ---
    model_path: str = ""
    image_height: int = 720
    image_width: int = 1080

    # Target FPS for the GrabberProcess. 10 FPS is enough for crowd counting
    # and saves PCIe bandwidth.
    fps: int = 10

    # Optional stream-count override (debugging). Unset → derive from topology.
    num_streams: Optional[int] = None
    num_buffers: int = 4
    num_workers_per_gpu: int = 1

    cuda_device: str = "0"

    # --- Source ---
    source_type: Literal["camera", "video"] = "camera"
    demo_video_path: Path = Path("../data/demo.mkv")

    # --- Handler enable flags ---
    # `enable_*` fields seed the *initial* state of runtime-toggleable
    # handlers; the GUI takes over once mounted. Non-toggleable handlers
    # (smart_plug, speaker) treat the flag as a hard on/off.
    enable_monitor: bool = False
    enable_video_recorder: bool = False
    enable_smart_plug: bool = True
    enable_speaker: bool = True

    tapo_email: str = ""
    tapo_password: str = ""

    # --- Alerting ---
    # A stream's count must remain above its threshold continuously for this
    # long before the alert fires (debounce, seconds).
    alert_trigger_delay: float = 5.0

    # --- Logging & recording ---
    # Audit log path (JSONL). Empty string disables auditing.
    audit_log_path: str = "logs/audit.jsonl"

    # Continuous video recording: every enabled stream is written to disk in
    # fixed-length segments. Toggleable per-stream at runtime from the debug GUI.
    video_record_dir: str = "recordings"
    video_segment_seconds: float = 300.0
    # cv2.VideoWriter fourcc. "mp4v" is stable everywhere out of the box.
    # Set to "avc1" for ~2-3x smaller H.264 files if a matching OpenH264 DLL
    # is on PATH (see https://github.com/cisco/openh264/releases). Falls
    # back to "mp4v" automatically if the configured codec can't open.
    video_fourcc: str = "mp4v"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _promote_verbose_to_debug(self) -> "EnvSettings":
        if self.verbose_debug:
            self.debug = True
        return self


# ======================================================================
# Master config (aggregates everything)
# ======================================================================


class Config:
    """Aggregates env settings, YAML topology, and runtime-derived fields
    (device, stream sources, sid mappings) into one immutable-ish object."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, envs: EnvSettings):
        self.envs = envs

        # 0. Up-front sanity checks (fail fast with actionable messages).
        self._validate_envs(envs)

        # 1. Build sub-configs.
        self._init_subconfigs()

        # 2. Detect hardware.
        self.device = self._detect_device(envs.cuda_device)

        # 3. Load topology + reconcile streams.
        self.zones = self._load_default_topology()
        self.stream_sources = self._resolve_stream_sources()

        # 4. Derived constants for downstream consumers.
        self.num_streams = len(self.stream_sources)
        self.num_buffers = self.shm.num_buffers
        self.num_workers_per_gpu = envs.num_workers_per_gpu
        self.fps = envs.fps
        self.frame_interval = 1.0 / envs.fps

        self._log_configuration()

    @classmethod
    def load(cls, envs: EnvSettings) -> "Config":
        """Factory: load configuration, exiting cleanly on any failure."""
        try:
            config = cls(envs)
            logger.info("Configuration Loaded Successfully.")
            return config
        except ConfigError as e:
            logger.critical("Configuration error: %s", e)
            sys.exit(1)
        except Exception as e:
            logger.critical("Failed to load configuration: %s", e, exc_info=True)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Validation & sub-config setup
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_envs(envs: EnvSettings) -> None:
        """Pre-flight validation. Raises ConfigError with actionable messages."""

        # 1. model_path is required; existence is a soft warning so container
        #    builds where weights are mounted at runtime still pass.
        if not envs.model_path:
            raise ConfigError("model_path is empty. Set MODEL_PATH in .env to point at your weights file.")
        mp_path = Path(envs.model_path)
        if not mp_path.exists():
            logger.warning("model_path=%s does not exist yet. Make sure it is mounted before launching.", mp_path)

        # 2. Demo-video mode requires the demo file.
        if envs.source_type == "video":
            demo_path = Path(envs.demo_video_path)
            if not demo_path.exists():
                raise ConfigError(
                    f"source_type=video but demo_video_path does not exist: {demo_path}. "
                    "Provide a valid mp4 path or switch source_type to 'camera'."
                )

        # 3. Topology file must exist (loaded later by `_load_default_topology`).
        yaml_path = Path(__file__).parents[1] / "topology.yaml"
        if not yaml_path.exists():
            raise ConfigError(
                f"Topology file not found: {yaml_path}. "
                "Create topology.yaml at the project root with at least one zone."
            )

    def _init_subconfigs(self) -> None:
        """Build the pydantic sub-config instances from `self.envs`."""
        envs = self.envs

        self.model = ModelConfig(path=Path(envs.model_path))

        self.shm = SHMSettings(
            num_buffers=envs.num_buffers,
            height=envs.image_height,
            width=envs.image_width,
        )

        self.plug_auth = SmartPlugAuthConfig(
            email=envs.tapo_email,
            password=envs.tapo_password,
        )

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device(device_id: str) -> torch.device:
        if torch.cuda.is_available():
            d = f"cuda:{device_id}"
            gpu_name = torch.cuda.get_device_name(int(device_id))
            logger.info("Hardware Accelerator: %s (%s)", gpu_name, d)
            return torch.device(d)
        logger.warning("Hardware Accelerator: CPU (Performance will be degraded)")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Topology & stream reconciliation
    # ------------------------------------------------------------------

    @staticmethod
    def _load_default_topology() -> tuple[ZoneConfig, ...]:
        """Parse topology.yaml. File existence is verified in `_validate_envs`."""
        yaml_path = Path(__file__).parents[1] / "topology.yaml"

        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"Failed to parse {yaml_path}: {e}") from e

        zones_raw = data.get("zones", [])
        if not zones_raw:
            raise ConfigError(
                f"{yaml_path} has no zones defined under the 'zones:' key. "
                "Define at least one zone with cameras/speakers/smart_plugs."
            )

        try:
            zones = tuple(ZoneConfig(**z) for z in zones_raw)
        except Exception as e:
            raise ConfigError(f"Invalid zone definition in {yaml_path}: {e}") from e

        logger.info("Successfully loaded %d zones from %s", len(zones), yaml_path)
        return zones

    def _resolve_stream_sources(self) -> tuple[str, ...]:
        """Flatten zones into a linear stream-source list.

        Camera mode → MJPEG URL per camera IP. Demo-video mode → the demo
        video path replicated for every camera slot (simulates full load).
        """
        sources: list[str] = []
        demo_url = str(self.envs.demo_video_path)
        for zone in self.zones:
            if not zone.cameras:
                continue
            if self.envs.source_type == "video":
                sources.extend([demo_url] * len(zone.cameras))
            else:
                sources.extend(self._camera_url(ip) for ip in zone.cameras)

        # Apply optional global limit (from .env).
        requested = self.envs.num_streams
        if requested is not None:
            if requested < len(sources):
                logger.warning("Limiting streams from %d to %d", len(sources), requested)
                return tuple(sources[:requested])
            if requested > len(sources):
                logger.warning(
                    "Requested %d streams but only found %d. Running with available sources.",
                    requested,
                    len(sources),
                )

        return tuple(sources)

    @staticmethod
    def _camera_url(ip: str) -> str:
        """Build the MJPEG stream URL for an AXIS-style IP camera."""
        return f"https://root:root@{ip}/mjpg/1/video.mjpg"

    # ------------------------------------------------------------------
    # Stream-id mappings (cached on first access)
    # ------------------------------------------------------------------

    @cached_property
    def sid_to_zone(self) -> dict[int, ZoneConfig]:
        return {sid: zone for sid, zone, _, _ in self._iter_camera_sids()}

    @cached_property
    def sid_to_threshold(self) -> dict[int, float]:
        # Debug mode pushes the threshold sky-high so no alert ever fires.
        debug_sentinel = 1e6
        return {
            sid: (debug_sentinel if self.envs.debug else float(threshold))
            for sid, _, _, threshold in self._iter_camera_sids()
        }

    @cached_property
    def sid_to_ip(self) -> dict[int, str]:
        return {sid: camera_ip for sid, _, camera_ip, _ in self._iter_camera_sids()}

    def _iter_camera_sids(self):
        """Yield `(sid, zone, camera_ip, threshold)` for every camera in zone order."""
        sid = 0
        for zone in self.zones:
            for camera_ip, threshold in zip(zone.cameras, zone.thresholds):
                yield sid, zone, camera_ip, threshold
                sid += 1

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_configuration(self) -> None:
        logger.info("=" * 40)
        logger.info("System Configuration Loaded")
        logger.info("Input Mode  : %s", self.envs.source_type)
        logger.info("Stream Count: %d", self.num_streams)
        logger.info("Target FPS  : %d", self.fps)
        logger.info("SHM Buffers : %d", self.num_buffers)
        logger.info("SHM Shape   : %dx%d", self.shm.height, self.shm.width)
        logger.info("Num Workers : %d", self.num_workers_per_gpu)
        logger.info("-" * 40)
        logger.info(">- Zone Topology")

        if not self.zones:
            logger.info("  [!] No zones configured.")
        else:
            for i, zone in enumerate(self.zones):
                logger.info("  + Zone %d: %s", i, zone.name)
                logger.info("    |-- Cameras    : %s", zone.cameras)
                logger.info("    |-- Speakers   : %s", zone.speakers)
                logger.info("    |-- SmartPlugs : %s", zone.smart_plugs)
                logger.info("    |-- Thresholds : %s", zone.thresholds)
                logger.info("")

        logger.info("=" * 40)
