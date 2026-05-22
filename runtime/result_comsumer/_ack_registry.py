import logging
import multiprocessing as mp
import queue
from collections.abc import Iterable
from typing import Optional

from runtime.shared_memory import BufferState, SharedMemory


logger = logging.getLogger(__name__)


class _PendingAckRegistry:
    """
    Tracks (sid, b_idx) buffers claimed by async handlers; releases them
    on ack arrival or after a timeout.

    SHM state-machine wise: handlers claim a buffer by including its
    (sid, b_idx) in the set they return from `handle_batch()`. The buffer
    stays in READING state until either `drain()` finds its ack on the
    queue or `sweep_stale()` force-releases it after `ack_timeout_sec`.
    """

    MAX_ACKS_PER_DRAIN = 128

    def __init__(
        self,
        ack_queue: Optional[mp.Queue],
        ack_timeout_sec: float,
        name: str = "ResultConsumer",
    ):
        self.ack_queue = ack_queue
        self.ack_timeout_sec = ack_timeout_sec
        self.name = name
        self.pending: dict[tuple[int, int], float] = {}

    def claim(self, pairs: Iterable[tuple[int, int]], now: float) -> None:
        """Mark a set of (sid, b_idx) buffers as awaiting ack."""
        for key in pairs:
            self.pending[key] = now

    def drain(self, shm: SharedMemory) -> None:
        """Non-blocking drain of incoming acks. Releases acked buffers."""
        if self.ack_queue is None:
            return
        released: list[tuple[int, int]] = []
        for _ in range(self.MAX_ACKS_PER_DRAIN):
            try:
                pairs = self.ack_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"[{self.name}] ack_queue read error: {e}")
                break
            for pair in pairs:
                key = (int(pair[0]), int(pair[1]))
                if self.pending.pop(key, None) is not None:
                    released.append(key)
        if released:
            self.release_buffers(shm, released)

    def sweep_stale(self, shm: SharedMemory, now: float) -> None:
        """Force-release buffers whose ack never arrived (handler crash safety net)."""
        if not self.pending:
            return
        cutoff = now - self.ack_timeout_sec
        stale = [k for k, t in self.pending.items() if t < cutoff]
        if not stale:
            return
        for k in stale:
            self.pending.pop(k, None)
        self.release_buffers(shm, stale)
        logger.warning(
            f"[{self.name}] Force-released {len(stale)} stale ack(s) after {self.ack_timeout_sec}s timeout."
        )

    def flush(self, shm: SharedMemory) -> None:
        """Release everything still pending. Call on shutdown."""
        if not self.pending:
            return
        self.release_buffers(shm, list(self.pending.keys()))
        self.pending.clear()

    @staticmethod
    def release_buffers(shm: SharedMemory, pairs) -> None:
        """Bulk-mark a list/set of (sid, b_idx) pairs FREE."""
        if not pairs:
            return
        sids = [p[0] for p in pairs]
        b_idxs = [p[1] for p in pairs]
        # `-1` is the sentinel for "no real buffer" used by inference workers
        # when they had nothing READY this tick; skip the SHM write entirely.
        if b_idxs[0] == -1:
            return
        shm.metadata[sids, b_idxs]["state"] = BufferState.FREE
