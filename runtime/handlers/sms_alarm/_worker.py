"""Off-thread I/O for the pile-up SMS alarm.

Everything in `alarm/` that touches disk or the network lives behind this
dispatcher, because the consumer thread cannot afford either:

* `EvidenceStore` opens/appends/closes `counts.csv` once per sample, and on an
  action also copies the snapshot and rewrites `pre_window_counts.csv`;
* `WorkerNotifier.notify` is a *blocking* `urlopen` with a 20 s timeout.

At 21 streams x 5 FPS a single stalled POST inside `handle_batch` would wedge
the whole result pipeline. So the handler keeps only the decision (motion filter
+ state machine, pure CPU, microseconds) inline and hands everything else here.

Two stages, deliberately:

    consumer thread ──put──> [queue] ──> evidence thread ──submit──> notify pool
                                        (disk, serial)              (HTTP, N-way)

The evidence thread stays serial so `counts.csv` rows keep their arrival order;
the notify pool is separate so one slow/hanging SMS POST cannot block evidence
writes for every other camera. `_evidence_lock` guards the `EvidenceStore`
because `record_notification` is called from pool threads.

Backpressure: a job that carries alarm actions is never dropped (it would mean a
silently missing SMS) — it waits briefly for room. Plain count samples are
dropped when the queue is full, which costs an evidence CSV row and nothing
else: the alarm decision for that sample has already been made upstream.
"""

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from alarm import AlarmAction, CountSample, EvidenceStore, WorkerNotifier
from alarm.time_utils import safe_name
from runtime.audit import AuditLog


logger = logging.getLogger(__name__)


# Depth of the consumer -> evidence-thread queue. Each entry is tiny (a frozen
# dataclass) unless it carries an alarm snapshot, and alarms are rare, so this
# is generous: ~5 s of backlog for 21 streams at 5 FPS.
QUEUE_MAXSIZE = 512

# How long an action-carrying job waits for queue room before we give up and
# log it as lost. Long enough to ride out a disk hiccup, short enough that a
# genuinely wedged evidence thread doesn't stall the consumer indefinitely.
ACTION_PUT_TIMEOUT = 2.0

# Concurrent in-flight SMS POSTs. Alarms fire on at most a handful of cameras at
# once; more threads would just queue up behind the same remote service.
NOTIFY_POOL_SIZE = 4

# Log every Nth dropped sample so a persistently full queue is visible without
# flooding the log during a transient stall.
DROP_LOG_EVERY = 100

# JPEG quality for evidence snapshots. 90 keeps pile-ups legible at ~150 KB,
# which matters because every snapshot is uploaded over the SMS worker API.
_JPEG_QUALITY = 90

# Shutdown budget for the evidence thread and for in-flight SMS sends.
_EVIDENCE_JOIN_TIMEOUT = 5.0
_NOTIFY_DRAIN_TIMEOUT = 25.0
_NOTIFY_DRAIN_POLL = 0.1


@dataclass(slots=True)
class AlarmJob:
    """One processed sample on its way to disk / the SMS worker.

    `event_id` is captured in the consumer thread rather than re-read here: the
    state machine keeps advancing, and by the time this job is picked up the
    event may already have closed.

    `frame` is a private copy of the SHM frame, taken only when `actions` is
    non-empty — the SHM slot is released as soon as `handle_batch` returns.
    """

    sample: CountSample
    event_id: Optional[str]
    actions: list[AlarmAction] = field(default_factory=list)
    frame: Optional[np.ndarray] = None


class AlarmDispatcher:
    """Owns the evidence thread, the notify pool and the alarm's disk/network I/O."""

    def __init__(
        self,
        evidence: EvidenceStore,
        notifier: WorkerNotifier,
        snapshot_dir: str | Path,
        name: str = "SmsAlarm",
    ):
        self.evidence = evidence
        self.notifier = notifier
        self.snapshot_dir = Path(snapshot_dir)
        self.name = name

        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pool: Optional[ThreadPoolExecutor] = None

        # Serializes EvidenceStore access: the evidence thread writes counts and
        # actions, pool threads append notification records.
        self._evidence_lock = threading.Lock()

        # In-flight notify futures, for a bounded drain on shutdown.
        self._inflight = 0
        self._inflight_lock = threading.Lock()

        self._drops = 0

        # Real AuditLog (or BaseHandler's _NullAudit stub) — handed over in start().
        self.audit: Optional[AuditLog] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, audit: AuditLog) -> None:
        self.audit = audit
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._pool = ThreadPoolExecutor(max_workers=NOTIFY_POOL_SIZE, thread_name_prefix=f"{self.name}-Notify")
        self._thread = threading.Thread(target=self._evidence_loop, name=f"{self.name}-Evidence", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)  # wake the loop out of its get()
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=_EVIDENCE_JOIN_TIMEOUT)
            if self._thread.is_alive():
                logger.warning("[%s] Evidence thread did not join in %.1fs.", self.name, _EVIDENCE_JOIN_TIMEOUT)
            self._thread = None
        self._drain_notifications()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if self._drops:
            logger.warning("[%s] Dropped %d count sample(s) from evidence.", self.name, self._drops)

    # ------------------------------------------------------------------
    # Producer side (called from the consumer thread)
    # ------------------------------------------------------------------

    def submit(self, job: AlarmJob) -> bool:
        """Queue one job. Returns False if it was dropped."""
        if self._thread is None:
            return False
        if job.actions:
            # Never drop an alarm: wait for room, and shout if we still lose it.
            try:
                self._queue.put(job, timeout=ACTION_PUT_TIMEOUT)
                return True
            except queue.Full:
                logger.error(
                    "[%s] Evidence queue full for %.1fs; LOST %d alarm action(s) for %s.",
                    self.name,
                    ACTION_PUT_TIMEOUT,
                    len(job.actions),
                    job.sample.camera_id,
                )
                self.audit.log(
                    "sms_alarm.action_lost",
                    camera_id=job.sample.camera_id,
                    actions=[a.action_type for a in job.actions],
                )
                return False
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self._drops += 1
            if self._drops == 1 or self._drops % DROP_LOG_EVERY == 0:
                logger.warning(
                    "[%s] Evidence queue full (drops: %d); count rows are being skipped.",
                    self.name,
                    self._drops,
                )
            return False

    # ------------------------------------------------------------------
    # Evidence thread
    # ------------------------------------------------------------------

    def _evidence_loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if job is None:
                return
            try:
                self._process(job)
            except Exception as e:
                logger.error("[%s] Evidence write failed: %s", self.name, e, exc_info=True)

    def _process(self, job: AlarmJob) -> None:
        sample = job.sample
        with self._evidence_lock:
            self.evidence.observe(sample)
            self.evidence.append_event_count(sample.camera_id, job.event_id, sample)

        for action in job.actions:
            snapshot = self._write_snapshot(action, job.frame)
            with self._evidence_lock:
                payload = self.evidence.record_action(
                    action,
                    replace(sample, frame_path=str(snapshot) if snapshot else None),
                )
            self._submit_notification(action, payload["snapshot_path"])

    def _write_snapshot(self, action: AlarmAction, frame: Optional[np.ndarray]) -> Optional[Path]:
        """Persist the alarm frame. None → `EvidenceStore` writes its placeholder."""
        if frame is None:
            return None
        path = self.snapshot_dir / (
            f"{safe_name(action.event_id)}_{safe_name(action.action_type)}_{int(action.timestamp)}.jpg"
        )
        try:
            ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        except Exception as e:
            logger.error("[%s] Snapshot encode failed for %s: %s", self.name, action.event_id, e, exc_info=True)
            return None
        if not ok:
            logger.error("[%s] cv2.imwrite refused to write %s.", self.name, path)
            return None
        return path

    # ------------------------------------------------------------------
    # Notify pool
    # ------------------------------------------------------------------

    def _submit_notification(self, action: AlarmAction, snapshot_path: str) -> None:
        if self._pool is None:
            return
        with self._inflight_lock:
            self._inflight += 1
        try:
            self._pool.submit(self._notify, action, snapshot_path)
        except RuntimeError:  # pool already shutting down
            with self._inflight_lock:
                self._inflight -= 1

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
                    "[%s] %s notification for %s was not sent: %s",
                    self.name,
                    action.action_type,
                    action.camera_id,
                    result.get("reason", result.get("status", "unknown")),
                )
        except Exception as e:
            logger.error("[%s] Notification failed for %s: %s", self.name, action.event_id, e, exc_info=True)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _drain_notifications(self) -> None:
        """Give in-flight SMS sends a bounded chance to finish before teardown."""
        deadline = time.monotonic() + _NOTIFY_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            with self._inflight_lock:
                if self._inflight <= 0:
                    return
            time.sleep(_NOTIFY_DRAIN_POLL)
        with self._inflight_lock:
            remaining = self._inflight
        if remaining > 0:
            logger.warning(
                "[%s] %d notification(s) still in flight after %.0fs drain; abandoning them.",
                self.name,
                remaining,
                _NOTIFY_DRAIN_TIMEOUT,
            )
