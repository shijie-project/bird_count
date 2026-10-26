"""SmsAlarmHandler — evidence + SMS for the chicken pile-up Level 1/2/3 alarm.

The decision itself (motion filter + Level 1/2/3 + recovery state machine) is
made once per sample by the consumer's alarm engine
(`runtime.result_comsumer._alarm_engine`) and arrives on each result as
`alarm_sample` / `alarm_actions` / `alarm_event_id`. This handler turns that
into the delivery package's evidence layout and SMS messages:

    handle_batch()  consumer thread   pick the frame for each action (SHM copy)
    AlarmDispatcher background        evidence CSV, snapshots, SMS POST

Every action is recorded as evidence. Only levels in `actions.sms.levels`
(configs/alarm.json, 1/2/3 by default) are texted, and a recovery is texted
only for an event this handler actually texted about — a "cleared" message for
an alarm nobody was told of would only confuse.

Enable with `ENABLE_SMS_ALARM=true`. SMS stays in dry-run (payloads and
snapshots written to disk, nothing sent) until `SMS_ALARM_REAL_WORKER=true` AND
`FARM_SMS_API_KEY` is exported.
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from alarm import EvidenceStore, WorkerNotifier, load_config
from runtime.config import Config
from runtime.handlers import BaseHandler
from runtime.inferencer import BatchInferenceResult, InferenceResult
from runtime.shared_memory import SharedMemory, SharedMemoryConfig

from ._worker import AlarmDispatcher, AlarmJob


logger = logging.getLogger(__name__)


class SmsAlarmHandler(BaseHandler):
    """Writes evidence and sends SMS for the alarm engine's actions.

    Results without an `alarm_sample` (streams the alarm config does not
    cover) are skipped; the engine reports those once at startup.
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
        self.dispatcher: Optional[AlarmDispatcher] = None
        self.levels: frozenset[int] = frozenset()

        self._config_path = Path(config.envs.alarm_config_path)
        self._output_dir_override = config.envs.alarm_output_dir or None

        # Open events (camera_id -> event_id), for the GUI status badges, and the
        # events an SMS went out for — the only ones whose recovery is texted.
        self._open_events: dict[str, str] = {}
        self._texted_events: set[str] = set()

    # ------------------------------------------------------------------
    # GUI surface (cancel-all + status badges, same slots as the speaker)
    # ------------------------------------------------------------------

    def cancel_all(self) -> None:
        """Forget open events. The engine drops them; no recovery will follow."""
        if self._open_events:
            self.audit.log("handler.cancel_all", handler=self.name, cameras=sorted(self._open_events))
        self._open_events.clear()
        self._texted_events.clear()

    def get_active_devices(self) -> set[str]:
        return set(self._open_events)

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

        self.levels = app_config.levels_for("sms")
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
            "[%s] Armed: SMS on level(s) %s + their recovery, output=%s, SMS=%s.",
            self.name,
            sorted(self.levels),
            output_dir,
            "REAL" if self._real_worker else "dry-run",
        )
        self.audit.log(
            "sms_alarm.start",
            config_path=str(self._config_path),
            output_dir=str(output_dir),
            real_worker=self._real_worker,
            levels=sorted(self.levels),
        )

    def stop(self) -> None:
        if self.dispatcher is not None:
            self.dispatcher.stop()
            self.dispatcher = None
        super().stop()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        # Unused — handle_batch is overridden.
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemory) -> set[tuple[int, int]]:
        """Queue every covered sample (and its actions) for evidence + SMS.

        Synchronous: returns an empty claim set, and any frame handed to the
        dispatcher is a private `.copy()`, so the consumer is free to release
        the SHM slot the moment this returns.
        """
        if not self._enabled or self.dispatcher is None:
            return set()

        frames = None  # resolved lazily — only an actual alarm needs pixels
        for result in batch_result.results:
            sample = result.alarm_sample
            if sample is None:
                continue
            try:
                actions = list(result.alarm_actions)
                frame = None
                if actions:
                    if frames is None:
                        frames = shm_client.frames
                    frame = frames[int(result.stream_id), int(result.buffer_idx)].copy()
                self.dispatcher.submit(
                    AlarmJob(
                        sample=sample,
                        event_id=result.alarm_event_id,
                        actions=actions,
                        frame=frame,
                        notify_types=self._track(actions),
                    )
                )
            except Exception as e:
                logger.error("[%s] Alarm evidence failed for %s: %s", self.name, sample.camera_id, e, exc_info=True)
        return set()

    def _track(self, actions) -> frozenset[str]:
        """Update open/texted events; return the action types to text."""
        notify = set()
        for action in actions:
            if action.action_type == "recovery":
                self._open_events.pop(action.camera_id, None)
                if action.event_id in self._texted_events:
                    self._texted_events.discard(action.event_id)
                    notify.add(action.action_type)
                continue
            self._open_events[action.camera_id] = action.event_id
            if action.level in self.levels:
                self._texted_events.add(action.event_id)
                notify.add(action.action_type)
        return frozenset(notify)
