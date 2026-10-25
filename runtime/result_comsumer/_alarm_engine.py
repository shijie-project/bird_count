"""_AlarmEngine — the one pile-up alarm decision, made once per sample.

Runs the vendored `alarm/` core (directional motion filter + Level 1/2/3 +
recovery state machine) over every result in a batch and writes the outcome
onto the result itself, *before* any handler sees it:

    result.alarm_actions   level1 / level2 / level3 / recovery, as emitted
    result.alarm_level     level of the open event, 0 when none
    result.alarm_sample    the CountSample the decision was made on
    result.alarm_event_id  the event open after this sample
    result.alert_flag      an event is open (or the stream is hijacked)

Handlers then *subscribe* to levels instead of each keeping its own rule: SMS
reacts to 1/2/3, the speaker and smart plug to whatever `actions` in
configs/alarm.json says (Level 1 by default). Recovery reaches every handler,
so anything switched on at some level is switched off when the event closes.

There is exactly one set of thresholds — `configs/alarm.json`, keyed by the
camera's MAC. A stream whose MAC is not in it gets no alarm at all; that is
reported once at startup.

Pure logic: no SHM, no GUI, no I/O on the hot path. Evidence files and SMS
sends stay in the SMS handler's background dispatcher.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from alarm import AlarmManager, CountSample, DirectionalMotionFilter, load_config
from alarm.camera_ids import resolve_camera_id
from runtime.audit import AuditLog
from runtime.config import Config
from runtime.inferencer import BatchInferenceResult, InferenceResult


logger = logging.getLogger(__name__)


class _AlarmEngine:
    """Motion filter + state machine for every covered stream.

    A config that cannot be loaded disables the engine (no stream alarms)
    rather than failing runtime construction; the error says why.
    """

    def __init__(self, config: Config, name: str = "AlarmEngine"):
        self.name = name
        self.enabled = False
        self.manager: AlarmManager | None = None
        self.motion_filter: DirectionalMotionFilter | None = None
        self.thresholds: dict[str, float] = {}
        # sid -> camera_id ("axisN/MAC"), covered streams only.
        self.sid_to_camera: dict[int, str] = {}

        config_path = Path(config.envs.alarm_config_path)
        try:
            app_config = load_config(config_path)
        except Exception as e:
            logger.error("[%s] Could not load alarm config %s: %s. No stream will alarm.", name, config_path, e)
            return

        unmapped: list[str] = []
        for sid in range(config.num_streams):
            mac = config.sid_to_mac.get(sid)
            camera_id = resolve_camera_id(mac=mac, known=app_config.cameras.keys())
            if camera_id is None:
                unmapped.append(f"sid={sid} ({config.sid_to_ip.get(sid, '?')}, mac={mac or 'unknown'})")
            else:
                self.sid_to_camera[sid] = camera_id

        self.thresholds = {cid: cam.threshold for cid, cam in app_config.cameras.items()}
        self.manager = AlarmManager(app_config.cameras, app_config.rules)
        self.motion_filter = DirectionalMotionFilter(app_config.motion_filter)
        self.enabled = bool(self.sid_to_camera)

        logger.info(
            "[%s] %d/%d stream(s) covered by %s. Action levels: %s.",
            name,
            len(self.sid_to_camera),
            config.num_streams,
            config_path,
            {h: sorted(lv) for h, lv in sorted(app_config.action_levels.items())},
        )
        if unmapped:
            logger.warning(
                "[%s] No alarm threshold for %d stream(s) — they will never alarm: %s",
                name,
                len(unmapped),
                ", ".join(unmapped),
            )

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def evaluate_inplace(
        self,
        batch_packet: BatchInferenceResult,
        manual_override_streams: set[int],
        audit: AuditLog,
    ) -> None:
        """Annotate every result with its alarm decision (see module docstring)."""
        for result in batch_packet.results:
            result.manual_override = result.stream_id in manual_override_streams
            camera_id = self.sid_to_camera.get(int(result.stream_id)) if self.enabled else None
            if camera_id is not None:
                try:
                    self._decide(result, camera_id, audit)
                except Exception as e:
                    logger.error("[%s] Alarm evaluation failed for %s: %s", self.name, camera_id, e, exc_info=True)
            result.alert_flag = result.manual_override or result.alarm_level > 0

    def _decide(self, result: InferenceResult, camera_id: str, audit: AuditLog) -> None:
        sample = self._build_sample(camera_id, result)
        actions = self.manager.process(sample)
        state = self.manager.state_of(camera_id)

        result.alarm_sample = sample
        result.alarm_actions = tuple(actions)
        result.alarm_event_id = state.event_id
        result.alarm_level = state.level if state.status != "normal" else 0

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
            audit.log(
                f"alarm.{action.action_type}",
                stream_id=int(result.stream_id),
                camera_id=camera_id,
                event_id=action.event_id,
                level=action.level,
                count=action.count,
                threshold=action.threshold,
            )

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

    # ------------------------------------------------------------------
    # Operator surface
    # ------------------------------------------------------------------

    def active_cameras(self) -> set[str]:
        return self.manager.active_cameras() if self.manager is not None else set()

    def reset_all(self, audit: AuditLog) -> Iterable[str]:
        """Cancel-all: drop every open event without a recovery action."""
        active = sorted(self.active_cameras())
        if self.manager is not None:
            self.manager.reset_all()
        if active:
            logger.info("[%s] cancel_all() dropped %d open event(s): %s", self.name, len(active), active)
            audit.log("alarm.cancel_all", cameras=active)
        return active
