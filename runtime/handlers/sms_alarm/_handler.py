"""SmsAlarmHandler — chicken pile-up Level 1/2/3 + recovery alarm over SMS.

Wraps the vendored `alarm/` core (see its module docstring for provenance) into
the runtime's handler contract. Replaces the delivery package's single-video
`model_runtime/inference.py`: counts come from this project's batched multi-
stream inferencer instead, so all cameras are covered by one GPU pass.

Split of work — the reason this is two files:

    handle_batch()  consumer thread   motion filter + state machine (pure CPU)
    AlarmDispatcher background        evidence CSV, snapshots, SMS POST

The decision half must be inline: the state machine measures "N >= T for 10
continuous seconds" against sample timestamps, so deferring it would distort the
timing. It is also cheap — a couple of dict lookups and some arithmetic per
sample. Everything with a filesystem or a socket in it goes to `_worker.py`.

Enable with `ENABLE_SMS_ALARM=true`. SMS stays in dry-run (payloads and
snapshots written to disk, nothing sent) until `SMS_ALARM_REAL_WORKER=true` AND
`FARM_SMS_API_KEY` is exported.

Thresholds come from `configs/alarm.json`, NOT topology.yaml — see the note on
`EnvSettings.alarm_config_path` for why the two are deliberately separate. Which
threshold applies to which stream is decided by the stream's bare MAC
(`Config.sid_to_mac`), translated to the config's `axisN/MAC` key by
`alarm.camera_ids`.
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from alarm import AlarmManager, CountSample, DirectionalMotionFilter, EvidenceStore, WorkerNotifier, load_config
from alarm.camera_ids import resolve_camera_id
from runtime.config import Config
from runtime.handlers import BaseHandler
from runtime.inferencer import BatchInferenceResult, InferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._worker import AlarmDispatcher, AlarmJob


logger = logging.getLogger(__name__)


class SmsAlarmHandler(BaseHandler):
    """Runs the pile-up alarm state machine over live inference results.

    Only streams that resolve to a camera id present in the alarm config are
    covered; the rest are listed once at startup and then ignored. That keeps a
    topology with more cameras than the alarm config (or vice versa) runnable
    instead of fatal.
    """

    needs_frames = False  # frames are resolved by hand, and only when an alarm fires

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        name: str = "SmsAlarm",
    ):
        super().__init__(config=config, shm_config=shm_config, name=name)

        self._enabled = bool(config.envs.enable_sms_alarm)
        self._real_worker = bool(config.envs.sms_alarm_real_worker)

        # Built in start(): the alarm config is read there so a bad file
        # disables this handler instead of killing runtime construction.
        self.manager: Optional[AlarmManager] = None
        self.motion_filter: Optional[DirectionalMotionFilter] = None
        self.dispatcher: Optional[AlarmDispatcher] = None
        self.thresholds: dict[str, float] = {}

        self._config_path = Path(config.envs.alarm_config_path)
        self._output_dir_override = config.envs.alarm_output_dir or None

        # sid -> camera_id, for covered streams only.
        self._sid_to_camera: dict[int, str] = {}
        self._unmapped: list[str] = []

    # ------------------------------------------------------------------
    # GUI surface (cancel-all + status badges, same slots as the speaker)
    # ------------------------------------------------------------------

    def cancel_all(self) -> None:
        """Drop every in-flight pile-up event without notifying."""
        if self.manager is None:
            return
        active = self.manager.active_cameras()
        if not active:
            return
        self.manager.reset_all()
        logger.info("[%s] cancel_all() reset %d active event(s): %s", self.name, len(active), sorted(active))
        self.audit.log("handler.cancel_all", handler=self.name, cameras=sorted(active))

    def get_active_devices(self) -> set[str]:
        return self.manager.active_cameras() if self.manager is not None else set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()
        if not self._enabled:
            return
        try:
            app_config = load_config(self._config_path)
        except Exception as e:
            logger.error(
                "[%s] Could not load alarm config %s: %s. Handler disabled.",
                self.name,
                self._config_path,
                e,
                exc_info=True,
            )
            self._enabled = False
            return

        self._resolve_stream_mapping(app_config.cameras)
        if not self._sid_to_camera:
            logger.error(
                "[%s] No stream resolved to a camera in %s. Handler disabled. "
                "Add bare-MAC `camera_ids:` entries to topology.yaml alongside `cameras:`.",
                self.name,
                self._config_path,
            )
            self._enabled = False
            return

        self.thresholds = {cid: cam.threshold for cid, cam in app_config.cameras.items()}
        self.manager = AlarmManager(app_config.cameras, app_config.rules)
        self.motion_filter = DirectionalMotionFilter(app_config.motion_filter)

        output_dir = Path(self._output_dir_override) if self._output_dir_override else app_config.output_dir
        evidence = EvidenceStore(output_dir, app_config.rules)
        # `enabled=True` mirrors the delivery bridge: the JSON's own `enabled`
        # flag is superseded by ENABLE_SMS_ALARM, and dry-run is driven by
        # SMS_ALARM_REAL_WORKER so the safe default can't be lost in a config file.
        worker_config = replace(app_config.worker, enabled=True, dry_run=not self._real_worker)
        notifier = WorkerNotifier(worker_config, mock_root=output_dir / "mock_worker")

        self.dispatcher = AlarmDispatcher(
            evidence=evidence,
            notifier=notifier,
            snapshot_dir=output_dir / "snapshots",
            name=self.name,
        )
        self.dispatcher.start(self.audit)

        logger.info(
            "[%s] Armed: %d/%d stream(s) covered, output=%s, SMS=%s.",
            self.name,
            len(self._sid_to_camera),
            self.config.num_streams,
            output_dir,
            "REAL" if self._real_worker else "dry-run",
        )
        if self._unmapped:
            logger.warning(
                "[%s] No alarm coverage for %d stream(s): %s",
                self.name,
                len(self._unmapped),
                ", ".join(self._unmapped),
            )
        self.audit.log(
            "sms_alarm.start",
            config_path=str(self._config_path),
            output_dir=str(output_dir),
            real_worker=self._real_worker,
            covered={str(sid): cid for sid, cid in sorted(self._sid_to_camera.items())},
            unmapped=self._unmapped,
        )

    def stop(self) -> None:
        if self.dispatcher is not None:
            self.dispatcher.stop()
            self.dispatcher = None
        super().stop()

    def _resolve_stream_mapping(self, cameras) -> None:
        """Populate `_sid_to_camera` / `_unmapped` from stream MACs + alarm config."""
        macs = self.config.sid_to_mac
        identifiers = self.config.sid_to_ip
        for sid in range(self.config.num_streams):
            mac = macs.get(sid)
            camera_id = resolve_camera_id(mac=mac, known=cameras.keys())
            if camera_id is None:
                self._unmapped.append(f"sid={sid} ({identifiers.get(sid, '?')}, mac={mac or 'unknown'})")
            else:
                self._sid_to_camera[sid] = camera_id

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        # Unused — handle_batch is overridden.
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemory) -> set[tuple[int, int]]:
        """Run the alarm decision for every covered stream in the batch.

        Synchronous: returns an empty claim set, and any frame handed to the
        dispatcher is a private `.copy()`, so the consumer is free to release
        the SHM slot the moment this returns.
        """
        if not self._enabled or self.manager is None or self.dispatcher is None:
            return set()

        frames = None  # resolved lazily — only an actual alarm needs pixels
        for result in batch_result.results:
            camera_id = self._sid_to_camera.get(int(result.stream_id))
            if camera_id is None:
                continue
            try:
                sample = self._build_sample(camera_id, result)
                actions = self.manager.process(sample)
                frame = None
                if actions:
                    if frames is None:
                        frames = shm_client.frames
                    frame = frames[int(result.stream_id), int(result.buffer_idx)].copy()
                    for action in actions:
                        logger.warning(
                            "[%s] %s on %s: N=%.1f T=%.1f (event %s)",
                            self.name,
                            action.action_type.upper(),
                            camera_id,
                            action.count,
                            action.threshold,
                            action.event_id,
                        )
                self.dispatcher.submit(
                    AlarmJob(
                        sample=sample,
                        event_id=self.manager.active_event_id(camera_id),
                        actions=actions,
                        frame=frame,
                    )
                )
            except Exception as e:
                logger.error("[%s] Alarm evaluation failed for %s: %s", self.name, camera_id, e, exc_info=True)
        return set()

    def _build_sample(self, camera_id: str, result: InferenceResult) -> CountSample:
        """Wrap one InferenceResult as a CountSample, applying the motion filter.

        `centroid_x/y` are read defensively: the inferencer does not emit density
        centroids yet. Without them `DirectionalMotionFilter` reports
        `missing_centroid` and the alarm degrades to a pure count rule — correct,
        just blind to whole-flock directional movement.
        """
        threshold = self.thresholds[camera_id]
        raw = CountSample(
            timestamp=float(result.timestamp),
            camera_id=camera_id,
            count=float(result.count),
            centroid_x=getattr(result, "centroid_x", None),
            centroid_y=getattr(result, "centroid_y", None),
        )
        motion = self.motion_filter.evaluate(raw, threshold)
        # Excluded samples are pushed just under the threshold rather than
        # zeroed, so the reported count stays truthful while the state machine
        # sees "not a pile-up". Same convention as the delivery bridge.
        decision_count = min(raw.count, threshold - 1e-3) if motion.excluded else raw.count
        return replace(
            raw,
            decision_count=decision_count,
            motion_excluded=motion.excluded,
            motion_reason=motion.reason,
            motion_velocity_x_norm_per_sec=motion.velocity_x_norm_per_sec,
            motion_velocity_y_norm_per_sec=motion.velocity_y_norm_per_sec,
            motion_speed_norm_per_sec=motion.speed_norm_per_sec,
            motion_net_displacement_norm=motion.net_displacement_norm,
            motion_direction_degrees=motion.direction_degrees,
            motion_direction_consistency=motion.direction_consistency,
        )
