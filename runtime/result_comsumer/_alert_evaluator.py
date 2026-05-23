import logging
import time

from runtime.audit import AuditLog
from runtime.inferencer import BatchInferenceResult


logger = logging.getLogger(__name__)


class _AlertEvaluator:
    """
    Threshold + hold-down evaluator with rising/falling-edge audit logging.

    Pure logic — no SHM, no GUI, no process state. Owns the alert state
    machine for every stream and emits one audit event per edge transition.
    Independently testable: construct with the threshold map + delay, call
    `evaluate_inplace` with batches.
    """

    def __init__(self, stream_thresholds: dict[int, float], trigger_delay: float):
        self.stream_thresholds = stream_thresholds
        self.trigger_delay = trigger_delay

        # Wall-clock time at which each stream's count first crossed its
        # threshold continuously. Cleared whenever the count drops below
        # threshold or the operator hits cancel-all.
        self.alert_first_seen: dict[int, float] = {}

        # Last observed alert state per stream — drives rising/falling-edge
        # audit emission.
        self._prev_alert: dict[int, bool] = {}

    def evaluate_inplace(
        self,
        batch_packet: BatchInferenceResult,
        manual_override_streams: set[int],
        audit: AuditLog,
    ) -> None:
        """
        Mutate each result's `alert_flag` in place and emit edge audit events.

        Avoids creating new objects to keep GC pressure low on the hot path.
        """
        now = time.time()
        for result in batch_packet.results:
            sid = result.stream_id
            threshold = self.stream_thresholds.get(sid, float("inf"))

            # 1. Manual hijack short-circuits everything (no hold-down).
            if sid in manual_override_streams:
                result.alert_flag = True
            elif result.count < threshold:
                # Drop below threshold -> reset hold-down timer.
                self.alert_first_seen.pop(sid, None)
                result.alert_flag = False
            else:
                # 2. Hold-down: alert only fires after sustained breach.
                first_seen = self.alert_first_seen.get(sid)
                if first_seen is None:
                    self.alert_first_seen[sid] = now
                    result.alert_flag = self.trigger_delay <= 0.0
                else:
                    result.alert_flag = (now - first_seen) >= self.trigger_delay

            # 3. Rising/falling-edge audit logging.
            prev = self._prev_alert.get(sid, False)
            if result.alert_flag and not prev:
                audit.log(
                    "alert.trigger",
                    stream_id=sid,
                    count=result.count,
                    threshold=threshold,
                    hijacked=sid in manual_override_streams,
                )
            elif prev and not result.alert_flag:
                audit.log("alert.clear", stream_id=sid, count=result.count)
            self._prev_alert[sid] = result.alert_flag

    def reset_holddown(self) -> None:
        """Cancel-all: clear hold-down + edge state so the next breach is logged fresh."""
        self.alert_first_seen.clear()
        self._prev_alert.clear()
