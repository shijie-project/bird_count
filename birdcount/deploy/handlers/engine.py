"""The pile-up alarm decision, made once per sample before any handler runs.

Runs the alarm core (directional motion filter + Level 1/2/3 + recovery state
machine) over every result from a camera the alarm config covers, and hands
handlers an immutable `Decision` per result. Handlers subscribe to levels
(`actions` in the alarm config) instead of keeping rules of their own, and
recovery reaches all of them, so whatever was switched on is switched off
when the event closes.

Thresholds come from exactly one place, the alarm config, keyed by the
camera's MAC; a camera not in it never alarms (reported once at startup).

Debug simulation: with DEBUG on and SMS in dry-run, a manual trigger makes
the camera *look* piled up (count lifted to 1.2 x its threshold), so a whole
event plays out through the real path: Level 1, escalation, evidence, dry-run
SMS, devices; un-trigger and recovery follows. Simulated samples carry
`motion_reason=manual_simulation`. With real SMS on, the trigger only switches
devices, so a stray click cannot text the farm.

Pure logic: no I/O, no frames.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from birdcount.deploy.alarm import AlarmManager, AppConfig, CountSample, DirectionalMotionFilter
from birdcount.deploy.alarm.camera_ids import resolve_camera_id
from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.base import Decision


logger = logging.getLogger(__name__)

# A simulated pile-up reports this multiple of the camera's threshold.
SIMULATED_COUNT_FACTOR = 1.2


class AlarmEngine:
    def __init__(
        self,
        config: AppConfig | None,
        macs: Sequence[str | None],
        *,
        simulate_manual: bool = False,
        audit: AuditLog = NULL_AUDIT,
    ):
        """`macs[sid]` is each camera's bare MAC; `config` None disables the alarm."""
        self.config = config
        self.audit = audit
        self.simulate_manual = simulate_manual
        self.camera_of: dict[int, str] = {}
        self.manager: AlarmManager | None = None
        self.motion: DirectionalMotionFilter | None = None
        if config is None:
            return
        for sid, mac in enumerate(macs):
            camera_id = resolve_camera_id(mac=mac, known=config.cameras.keys())
            if camera_id is not None:
                self.camera_of[sid] = camera_id
        self.manager = AlarmManager(config.cameras, config.rules)
        self.motion = DirectionalMotionFilter(config.motion_filter)
        logger.info(
            "Alarm covers %d/%d camera(s). Action levels: %s.",
            len(self.camera_of),
            len(macs),
            {h: sorted(levels) for h, levels in sorted(config.action_levels.items())},
        )
        uncovered = [f"#{sid} ({mac or 'no MAC'})" for sid, mac in enumerate(macs) if sid not in self.camera_of]
        if uncovered:
            logger.warning(
                "No alarm threshold for %d camera(s); they never alarm: %s", len(uncovered), ", ".join(uncovered)
            )

    def threshold_of(self, sid: int) -> float | None:
        camera_id = self.camera_of.get(sid)
        return None if camera_id is None else self.config.cameras[camera_id].threshold

    def levels_for(self, handler: str) -> frozenset[int]:
        return self.config.levels_for(handler) if self.config is not None else frozenset()

    # ------------------------------------------------------------------

    def evaluate(self, records: np.ndarray, hijacked: set[int]) -> tuple[list[Decision | None], frozenset[int]]:
        """Decisions aligned with `records`, and the cameras switched on by hand.

        A hijacked camera is simulated through the alarm when simulation is on
        and the camera is covered; otherwise it is a plain device-only override.
        """
        decisions: list[Decision | None] = []
        manual = set()
        # Plain lists: field access on numpy records costs ~1 us each, per frame.
        for sid, ts, count, cx, cy in zip(
            *(records[k].tolist() for k in ("sid", "captured", "count", "cx", "cy")), strict=True
        ):
            camera_id = self.camera_of.get(sid)
            simulated = sid in hijacked and self.simulate_manual and camera_id is not None
            if sid in hijacked and not simulated:
                manual.add(sid)
            decision = None
            if camera_id is not None:
                try:
                    decision = self._decide(sid, ts, count, cx, cy, camera_id, simulated)
                except Exception:
                    logger.exception("Alarm evaluation failed for %s.", camera_id)
            decisions.append(decision)
        return decisions, frozenset(manual)

    def _decide(
        self, sid: int, ts: float, count: float, cx: float, cy: float, camera_id: str, simulated: bool
    ) -> Decision:
        sample = self._sample(ts, count, cx, cy, camera_id, simulated)
        actions = tuple(self.manager.process(sample))
        state = self.manager.state_of(camera_id)
        for a in actions:
            logger.warning(
                "%s on %s: N=%.1f T=%.1f (event %s)",
                a.action_type.upper(),
                camera_id,
                a.count,
                a.threshold,
                a.event_id,
            )
            self.audit.log(
                f"alarm.{a.action_type}",
                stream_id=sid,
                camera_id=camera_id,
                event_id=a.event_id,
                level=a.level,
                count=a.count,
                threshold=a.threshold,
                simulated=simulated,
            )
        return Decision(
            camera_id=camera_id,
            sample=sample,
            actions=actions,
            level=state.level if state.status != "normal" else 0,
            event_id=state.event_id,
            simulated=simulated,
        )

    def _sample(
        self, timestamp: float, count: float, cx: float, cy: float, camera_id: str, simulated: bool
    ) -> CountSample:
        """One result as a CountSample, through the motion filter (or the simulation)."""
        threshold = self.config.cameras[camera_id].threshold
        if math.isnan(cx):
            cx = cy = None
        if simulated:
            count = max(count, SIMULATED_COUNT_FACTOR * threshold)
            return CountSample(
                timestamp=timestamp,
                camera_id=camera_id,
                count=count,
                centroid_x=cx,
                centroid_y=cy,
                decision_count=count,
                motion_reason="manual_simulation",
            )
        raw = CountSample(timestamp=timestamp, camera_id=camera_id, count=count, centroid_x=cx, centroid_y=cy)
        motion = self.motion.evaluate(raw, threshold)
        # Excluded samples sit just under the threshold rather than at zero: the
        # reported count stays truthful while the state machine sees no pile-up.
        return replace(
            raw,
            decision_count=min(count, threshold - 1e-3) if motion.excluded else count,
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

    def active_cameras(self) -> set[str]:
        return self.manager.active_cameras() if self.manager is not None else set()

    def reset_all(self) -> list[str]:
        """Cancel-all: drop every open event, without recovery actions."""
        active = sorted(self.active_cameras())
        if self.manager is not None:
            self.manager.reset_all()
        if active:
            logger.info("Cancel-all dropped %d open event(s): %s", len(active), active)
            self.audit.log("alarm.cancel_all", cameras=active)
        return active
