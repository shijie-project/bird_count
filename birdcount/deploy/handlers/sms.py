"""Evidence and SMS for the pile-up alarm.

Turns the engine's decisions into the delivery package's evidence layout and
SMS messages. Every action is recorded as evidence; only levels in
`actions.sms.levels` (1/2/3 by default) are texted, and a recovery is texted
only for an event this handler actually texted about — a "cleared" message
for an alarm nobody was told of would only confuse.

The consumer thread keeps only the bookkeeping and the snapshot copy;
everything else runs off-thread in two stages:

    consumer --put--> [queue] --> evidence thread --submit--> notify pool
                                 (disk, serial)               (HTTP, N-way)

The evidence thread stays serial so counts.csv keeps arrival order; the
notify pool is separate so one hanging POST cannot hold up evidence for
every other camera. A job carrying alarm actions is never dropped (it waits
briefly for room and shouts if it still cannot get in); plain count rows are
dropped on a full queue, which loses a CSV row and nothing else.

SMS stay in dry-run (payloads and snapshots written to disk, nothing sent)
unless SMS_ALARM_REAL_WORKER is on and FARM_SMS_API_KEY is set.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from birdcount.deploy.alarm import AlarmAction, AppConfig, CountSample, EvidenceStore, WorkerNotifier
from birdcount.deploy.alarm.time_utils import snapshot_name
from birdcount.deploy.audit import NULL_AUDIT, AuditLog
from birdcount.deploy.handlers.base import Batch, Handler


logger = logging.getLogger(__name__)

QUEUE_MAXSIZE = 512  # ~5 s of backlog for 20 cameras at 5 fps
ACTION_PUT_TIMEOUT = 2.0  # how long an alarm job waits for queue room before it is reported lost
NOTIFY_POOL_SIZE = 4  # concurrent SMS POSTs; alarms fire on a handful of cameras at most
DROP_LOG_EVERY = 100
JPEG_QUALITY = 90  # evidence snapshots: legible pile-ups at ~150 KB, uploaded with the SMS
_EVIDENCE_JOIN_TIMEOUT = 5.0
_NOTIFY_DRAIN_TIMEOUT = 25.0


@dataclass(slots=True)
class AlarmJob:
    """One sample on its way to disk / the SMS worker.

    `event_id` is captured on the consumer thread: by the time the job is
    processed the state machine may have moved on. `frame` is a private copy,
    taken only when there are actions. Actions are all recorded; those in
    `notify_types` are also texted.
    """

    sample: CountSample
    event_id: str | None
    actions: tuple[AlarmAction, ...] = ()
    frame: np.ndarray | None = None
    notify_types: frozenset[str] = frozenset()


class AlarmDispatcher:
    """The evidence thread, the notify pool, and all of the alarm's disk and network I/O."""

    def __init__(self, evidence: EvidenceStore, notifier: WorkerNotifier, snapshot_dir: Path, audit: AuditLog):
        self.evidence = evidence
        self.notifier = notifier
        self.snapshot_dir = snapshot_dir
        self.audit = audit
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._evidence_lock = threading.Lock()  # evidence thread writes counts/actions, pool threads notifications
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self.drops = 0

    def start(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=NOTIFY_POOL_SIZE, thread_name_prefix="SmsNotify")
        self._thread = threading.Thread(target=self._evidence_loop, name="SmsEvidence", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)  # blocking: the sentinel lands behind every queued job
        self._thread.join(timeout=_EVIDENCE_JOIN_TIMEOUT)
        if self._thread.is_alive():
            logger.warning("SMS evidence thread did not finish within %.0f s.", _EVIDENCE_JOIN_TIMEOUT)
        self._thread = None
        self._drain_notifications()
        self._pool.shutdown(wait=False)
        self._pool = None
        if self.drops:
            logger.warning("Dropped %d count sample(s) from the alarm evidence.", self.drops)

    def submit(self, job: AlarmJob) -> bool:
        """Queue a job; False if it was dropped (or, for an alarm, lost)."""
        if self._thread is None:
            return False
        if job.actions:
            try:
                self._queue.put(job, timeout=ACTION_PUT_TIMEOUT)
                return True
            except queue.Full:
                types = [a.action_type for a in job.actions]
                logger.error(
                    "Evidence queue full for %.0f s: LOST %s for %s.", ACTION_PUT_TIMEOUT, types, job.sample.camera_id
                )
                self.audit.log("sms_alarm.action_lost", camera_id=job.sample.camera_id, actions=types)
                return False
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self.drops += 1
            if self.drops == 1 or self.drops % DROP_LOG_EVERY == 0:
                logger.warning("Evidence queue full (drops: %d); count rows are being skipped.", self.drops)
            return False

    # Evidence thread -------------------------------------------------------

    def _evidence_loop(self) -> None:
        while (job := self._queue.get()) is not None:
            try:
                self._process(job)
            except Exception:
                logger.exception("Evidence write failed.")

    def _process(self, job: AlarmJob) -> None:
        sample = job.sample
        with self._evidence_lock:
            self.evidence.observe(sample)
            self.evidence.append_event_count(sample.camera_id, job.event_id, sample)
        for action in job.actions:
            snapshot = self._write_snapshot(action, job.frame)
            with self._evidence_lock:
                payload = self.evidence.record_action(
                    action, replace(sample, frame_path=str(snapshot) if snapshot else None)
                )
            if action.action_type in job.notify_types:
                self._submit_notification(action, payload["snapshot_path"])

    def _write_snapshot(self, action: AlarmAction, frame: np.ndarray | None) -> Path | None:
        """The alarm frame as JPEG; None lets the evidence store draw its placeholder."""
        if frame is None:
            return None
        path = self.snapshot_dir / snapshot_name(action.event_id, action.action_type, action.timestamp)
        if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
            logger.error("Could not write snapshot %s.", path)
            return None
        return path

    # Notify pool --------------------------------------------------------------

    def _submit_notification(self, action: AlarmAction, snapshot_path: str) -> None:
        with self._inflight_lock:
            self._inflight += 1
        self._pool.submit(self._notify, action, snapshot_path)

    def _notify(self, action: AlarmAction, snapshot_path: str) -> None:
        try:
            result = self.notifier.notify(action, snapshot_path)
            with self._evidence_lock:
                self.evidence.record_notification(action, result)
            self.audit.log(
                "sms_alarm.notify",
                camera_id=action.camera_id,
                event_id=action.event_id,
                action_type=action.action_type,
                level=action.level,
                count=action.count,
                threshold=action.threshold,
                sent=bool(result.get("sent")),
                dry_run=bool(result.get("dry_run", False)),
                reason=result.get("reason", ""),
            )
            if not result.get("sent"):
                logger.error(
                    "%s SMS for %s was not sent: %s",
                    action.action_type,
                    action.camera_id,
                    result.get("reason", result.get("status", "unknown")),
                )
        except Exception:
            logger.exception("SMS notification failed for %s.", action.event_id)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _drain_notifications(self) -> None:
        """Give in-flight SMS a bounded chance to finish before teardown."""
        deadline = time.monotonic() + _NOTIFY_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            with self._inflight_lock:
                if self._inflight <= 0:
                    return
            time.sleep(0.1)
        logger.warning("%d SMS still in flight after %.0f s; abandoning them.", self._inflight, _NOTIFY_DRAIN_TIMEOUT)


class SmsHandler(Handler):
    name = "SMS"

    def __init__(
        self,
        config: AppConfig,
        *,
        output_dir: Path | None = None,
        real_worker: bool = False,
        api_key: str = "",
        audit: AuditLog = NULL_AUDIT,
    ):
        super().__init__(audit)
        self.levels = config.levels_for("sms")
        self.output_dir = output_dir or config.output_dir
        # The config's own worker `enabled` / `dry_run` flags are superseded by
        # ENABLE_SMS_ALARM / SMS_ALARM_REAL_WORKER, so a config file cannot turn
        # off the safe dry-run default.
        worker = replace(config.worker, enabled=True, dry_run=not real_worker)
        self.real_worker = real_worker
        self.dispatcher = AlarmDispatcher(
            evidence=EvidenceStore(self.output_dir, config.rules),
            notifier=WorkerNotifier(worker, mock_root=self.output_dir / "mock_worker", api_key=api_key),
            snapshot_dir=self.output_dir / "snapshots",
            audit=audit,
        )
        self._open_events: dict[str, str] = {}  # camera_id -> event_id, for the status badges
        self._texted_events: set[str] = set()  # events an SMS went out for: only their recovery is texted

    def start(self) -> None:
        self.dispatcher.start()
        logger.info(
            "SMS armed on level(s) %s + recovery; output %s; %s.",
            sorted(self.levels),
            self.output_dir,
            "REAL SMS" if self.real_worker else "dry-run",
        )
        self.audit.log(
            "sms_alarm.start",
            output_dir=str(self.output_dir),
            real_worker=self.real_worker,
            levels=sorted(self.levels),
        )

    def stop(self) -> None:
        self.dispatcher.stop()

    def handle(self, batch: Batch) -> None:
        for i, decision in enumerate(batch.decisions):
            if decision is None:
                continue
            self.dispatcher.submit(
                AlarmJob(
                    sample=decision.sample,
                    event_id=decision.event_id,
                    actions=decision.actions,
                    frame=batch.frame(i).copy() if decision.actions else None,
                    notify_types=self._track(decision.actions),
                )
            )

    def _track(self, actions: tuple[AlarmAction, ...]) -> frozenset[str]:
        """Update open / texted events; return the action types to text."""
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

    def cancel_all(self) -> None:
        """Forget open events: the engine drops them, so no recovery will follow."""
        if self._open_events:
            self.audit.log("handler.cancel_all", handler=self.name, cameras=sorted(self._open_events))
        self._open_events.clear()
        self._texted_events.clear()

    def active_devices(self) -> set[str]:
        return set(self._open_events)
