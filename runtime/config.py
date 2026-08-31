"""Configuration root.

`EnvSettings` reads from `.env` / system env vars via pydantic-settings.
`Config` wraps that input plus the YAML topology and runtime-derived fields
(device, stream sources, sid mappings) into a single source of truth that the
rest of `runtime/` reads from.
"""

import logging
import re
import sys
from functools import cached_property
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

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


# File extensions treated as video files when a directory is given as a source.
_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".webm",
    ".flv",
    ".wmv",
    ".ts",
}

# Threshold / zone applied to video streams that have no matching camera slot
# in topology.yaml (i.e. more videos were supplied than there are cameras).
_DEFAULT_VIDEO_THRESHOLD = 60.0


class StreamSpec(NamedTuple):
    """One capture stream, resolved from either a camera IP or a video file.

    `source` is what `CameraThread` opens (an MJPEG URL or a file path);
    `identifier` is the human-facing name shown in the GUI / monitor (camera
    IP or video filename); `zone` / `threshold` drive alerting and IoT routing.
    `camera_id` is the optional `axisN/MAC` id the pile-up alarm config uses
    (`configs/alarm.json`), taken verbatim from topology.yaml when present.
    """

    source: str
    identifier: str
    zone: "ZoneConfig"
    threshold: float
    camera_id: Optional[str] = None


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

    # Optional `axisN/MAC` ids for the pile-up SMS alarm, positionally aligned
    # with `cameras`. Left empty the alarm handler falls back to inferring the
    # id from the stream source path / identifier (see `alarm.camera_ids`).
    camera_ids: list[str] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def _pad_camera_ids(self) -> "ZoneConfig":
        """Pad `camera_ids` with empty strings so it can be zipped with `cameras`.

        Partial mapping is legitimate: only some cameras may exist in the alarm
        config, and the rest simply get no SMS alarm coverage.
        """
        if len(self.camera_ids) > len(self.cameras):
            raise ValueError(
                f"zone {self.name!r} lists {len(self.camera_ids)} camera_ids but only {len(self.cameras)} cameras"
            )
        self.camera_ids = list(self.camera_ids) + [""] * (len(self.cameras) - len(self.camera_ids))
        return self


class SmartPlugAuthConfig(BaseModel):
    """Credentials for external device control. Sourced from `EnvSettings.tapo_*`."""

    email: str = ""
    password: str = ""


class SpeakerAuthConfig(BaseModel):
    """Basic-auth credentials for the network speakers' /cgi-bin API.

    Defaults are the factory ones — override via `EnvSettings.speaker_*` on any
    deployment where the speakers have been re-provisioned.
    """

    username: str = "admin"
    password: str = "admin"


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

    # Single-video fallback (backward compatible). Used only when
    # `demo_video_paths` below is empty. May point at a file OR a directory
    # (a directory expands to all video files it contains, sorted by name).
    demo_video_path: Path = Path("../data/demo.mkv")

    # Multiple videos: a comma / semicolon / newline-separated list of video
    # files and/or directories. Each file becomes one stream ("N videos → N
    # streams"); each directory expands to the video files inside it (sorted).
    # When set, this takes precedence over `demo_video_path`.
    demo_video_paths: str = ""

    # --- Handler enable flags ---
    # `enable_*` fields seed the *initial* state of runtime-toggleable
    # handlers; the GUI takes over once mounted. Non-toggleable handlers
    # (smart_plug, speaker) treat the flag as a hard on/off.
    enable_monitor: bool = False
    enable_video_recorder: bool = False
    enable_smart_plug: bool = True
    enable_speaker: bool = True
    enable_sms_alarm: bool = False

    show_density_map: bool = False

    tapo_email: str = ""
    tapo_password: str = ""

    # Speaker credentials + the deterrent clip to broadcast. The clip must
    # already be uploaded to every speaker under this exact name.
    speaker_username: str = "admin"
    speaker_password: str = "admin"
    speaker_audio_file: str = "7MB.wav"

    # --- Alerting ---
    # A stream's count must remain above its threshold continuously for this
    # long before the alert fires (debounce, seconds).
    alert_trigger_delay: float = 5.0

    # --- Pile-up SMS alarm (runtime/handlers/sms_alarm) ---
    # Level 1/2/3 + recovery state machine, evidence capture and SMS dispatch.
    # Its thresholds live in `alarm_config_path`, NOT in topology.yaml: the
    # topology `thresholds:` drive speaker/smart-plug deterrence, a different
    # policy with different numbers. Note `debug=True` mutes the deterrence
    # alert (see `sid_to_threshold`) but deliberately does NOT mute this alarm.
    alarm_config_path: str = "configs/alarm.json"

    # Empty → use the `output_dir` recorded inside `alarm_config_path`.
    alarm_output_dir: str = ""

    # False (default) routes SMS through the delivery package's dry-run path:
    # real payloads and snapshots are written to disk, nothing is sent. Set to
    # True *and* export FARM_SMS_API_KEY to send for real.
    sms_alarm_real_worker: bool = False

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
        self._stream_specs = self._build_stream_specs()
        self.stream_sources = tuple(spec.source for spec in self._stream_specs)

        # 4. Derived constants for downstream consumers.
        self.num_streams = len(self._stream_specs)
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

        # 2. Demo-video mode requires at least one existing video file.
        if envs.source_type == "video":
            videos = Config._resolve_video_paths(envs)
            if not videos:
                raise ConfigError(
                    "source_type=video but no video files were resolved. Set DEMO_VIDEO_PATHS "
                    "(comma-separated files/directories) or DEMO_VIDEO_PATH, or switch "
                    "source_type to 'camera'."
                )
            missing = [str(v) for v in videos if not v.exists()]
            if missing:
                raise ConfigError(
                    "source_type=video but these video files do not exist: "
                    f"{missing}. Fix the paths or switch source_type to 'camera'."
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

        self.speaker_auth = SpeakerAuthConfig(
            username=envs.speaker_username,
            password=envs.speaker_password,
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

    def _build_stream_specs(self) -> tuple[StreamSpec, ...]:
        """Resolve the linear list of capture streams.

        Camera mode → one stream per camera IP, in zone order. Video mode →
        one stream per resolved video file ("N videos → N streams"), each
        borrowing the zone / threshold of the matching camera slot when one
        exists (so per-zone alerting/IoT still applies), and falling back to
        a default zone/threshold for any extra videos.
        """
        if self.envs.source_type == "video":
            specs = self._build_video_specs()
        else:
            specs = self._build_camera_specs()

        # Apply optional global limit (from .env).
        requested = self.envs.num_streams
        if requested is not None:
            if requested < len(specs):
                logger.warning("Limiting streams from %d to %d", len(specs), requested)
                specs = specs[:requested]
            elif requested > len(specs):
                logger.warning(
                    "Requested %d streams but only found %d. Running with available sources.",
                    requested,
                    len(specs),
                )

        return tuple(specs)

    def _build_camera_specs(self) -> list[StreamSpec]:
        """One StreamSpec per camera IP, flattened across zones in order."""
        return [
            StreamSpec(
                source=self._camera_url(ip),
                identifier=ip,
                zone=zone,
                threshold=threshold,
                camera_id=camera_id,
            )
            for zone, ip, threshold, camera_id in self._iter_topology_slots()
        ]

    def _build_video_specs(self) -> list[StreamSpec]:
        """One StreamSpec per resolved video file.

        Videos are matched to topology camera slots by position, so the i-th
        video inherits the i-th camera's zone + threshold. Videos beyond the
        number of camera slots get a synthetic default zone (no IoT devices)
        and the default threshold — the demo still runs, just without routing.
        """
        videos = self._resolve_video_paths(self.envs)
        slots = list(self._iter_topology_slots())
        default_zone = ZoneConfig(name="video")

        specs: list[StreamSpec] = []
        for i, video in enumerate(videos):
            if i < len(slots):
                zone, _, threshold, _ = slots[i]
            else:
                zone, threshold = default_zone, _DEFAULT_VIDEO_THRESHOLD
            specs.append(
                StreamSpec(
                    source=str(video),
                    identifier=video.stem,
                    zone=zone,
                    threshold=threshold,
                    # Deliberately not `camera_id`: the positional topology slot
                    # says nothing about which camera a video came from. The alarm
                    # handler infers the id from the file path instead.
                    camera_id=None,
                )
            )
        return specs

    def _iter_topology_slots(self):
        """Yield `(zone, camera_ip, threshold, camera_id)` for every camera in zone order."""
        for zone in self.zones:
            for camera_ip, threshold, camera_id in zip(zone.cameras, zone.thresholds, zone.camera_ids):
                yield zone, camera_ip, float(threshold), (camera_id or None)

    @staticmethod
    def _resolve_video_paths(envs: EnvSettings) -> list[Path]:
        """Expand the configured video source(s) into a concrete file list.

        Precedence: `demo_video_paths` (a comma/semicolon/newline-separated
        list) when set, else the single `demo_video_path`. Each entry may be a
        file (used as-is) or a directory (expanded to its video files, sorted).
        """
        raw = envs.demo_video_paths.strip()
        if raw:
            entries = [e.strip() for e in re.split(r"[,;\n]", raw) if e.strip()]
        else:
            entries = [str(envs.demo_video_path)]

        paths: list[Path] = []
        for entry in entries:
            p = Path(entry)
            if p.is_dir():
                paths.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in _VIDEO_EXTENSIONS))
            else:
                paths.append(p)
        return paths

    @staticmethod
    def _camera_url(ip: str) -> str:
        """Build the MJPEG stream URL for an AXIS-style IP camera."""
        return f"https://root:root@{ip}/mjpg/1/video.mjpg"

    # ------------------------------------------------------------------
    # Stream-id mappings (cached on first access)
    # ------------------------------------------------------------------

    @cached_property
    def sid_to_zone(self) -> dict[int, ZoneConfig]:
        return {sid: spec.zone for sid, spec in enumerate(self._stream_specs)}

    @cached_property
    def sid_to_threshold(self) -> dict[int, float]:
        # Debug mode pushes the threshold sky-high so no alert ever fires.
        debug_sentinel = 1e6
        return {
            sid: (debug_sentinel if self.envs.debug else spec.threshold) for sid, spec in enumerate(self._stream_specs)
        }

    @cached_property
    def sid_to_ip(self) -> dict[int, str]:
        return {sid: spec.identifier for sid, spec in enumerate(self._stream_specs)}

    @cached_property
    def sid_to_source(self) -> dict[int, str]:
        return {sid: spec.source for sid, spec in enumerate(self._stream_specs)}

    @cached_property
    def sid_to_camera_id(self) -> dict[int, Optional[str]]:
        """Explicit `axisN/MAC` alarm ids from topology.yaml (None where unset)."""
        return {sid: spec.camera_id for sid, spec in enumerate(self._stream_specs)}

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
