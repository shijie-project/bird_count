"""Inference process: frame ring -> `core.inference.DensityPredictor` -> results.

Runs on the ring's clock. At each tick it waits until every camera it serves
has a frame newer than the last one it processed (or a deadline passes), then
infers them as one batch. The grabber publishes the first frame decoded after
each tick, so a batch fills within one camera frame period and a frame's
latency is that wait plus one batch — instead of up to a whole tick, as when
v2's grabber and inferencer each ran on their own unsynchronised schedule.

Cameras that have produced nothing new for a few ticks (disconnected) are
not waited for, so one dead camera cannot delay the others.

Only this process loads torch; importing this module does not.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from birdcount.core.log import setup_logging
from birdcount.deploy.results import empty_batch
from birdcount.deploy.shm import FrameRing, RingLayout


if TYPE_CHECKING:
    from birdcount.core.inference import DensityPredictor


logger = logging.getLogger(__name__)

_POLL = 0.001  # s between checks for fresh frames
_STALLED_TICKS = 3  # a camera silent this many ticks is not waited for
_QUEUE_TIMEOUT = 0.05


class InferenceLoop:
    """One worker's batching logic, separate from the process for testing."""

    def __init__(
        self,
        ring: FrameRing,
        predictor: DensityPredictor,
        sids: Sequence[int],
        *,
        mask_rows: Sequence[int] | None = None,
        max_wait: float | None = None,
    ):
        if len(sids) > predictor.max_batch:
            raise ValueError(f"{len(sids)} cameras but the predictor batches at most {predictor.max_batch}")
        self.ring = ring
        self.predictor = predictor
        self.sids = list(sids)
        self.mask_rows = None if mask_rows is None else list(mask_rows)
        self.max_wait = min(0.5 * ring.interval, 0.1) if max_wait is None else max_wait
        self.last_seen = dict.fromkeys(self.sids, 0)  # frame_idx last inferred per camera
        self.last_fresh = dict.fromkeys(self.sids, time.monotonic())

    def _fresh(self) -> dict[int, tuple[int, int]]:
        out = {}
        for sid in self.sids:
            newest = self.ring.newest_ready(sid)
            if newest is not None and newest[1] > self.last_seen[sid]:
                out[sid] = newest
        return out

    def gather(self, deadline: float) -> dict[int, tuple[int, int]]:
        """Fresh (slot, frame_idx) per camera, once all live cameras have one or at the deadline."""
        stalled_after = _STALLED_TICKS * self.ring.interval
        while True:
            now = time.monotonic()
            fresh = self._fresh()
            for sid in fresh:
                self.last_fresh[sid] = now
            live = [s for s in self.sids if now - self.last_fresh[s] < stalled_after]
            if (fresh and all(s in fresh for s in live)) or now >= deadline:
                return fresh
            time.sleep(_POLL)

    def step(self, deadline: float) -> np.ndarray | None:
        """Infer the next batch; returns its result records, or None when nothing was fresh."""
        fresh = self.gather(deadline)
        if not fresh:
            return None
        ring, staging = self.ring, self.predictor.staging
        taken: list[tuple[int, int, int]] = []
        for sid, (slot, frame_idx) in fresh.items():
            ring.begin_read(sid, slot)
            np.copyto(staging[len(taken)], ring.frames[sid, slot])
            if ring.verify(sid, slot, frame_idx):
                taken.append((sid, slot, frame_idx))
            self.last_seen[sid] = max(self.last_seen[sid], frame_idx)
        if not taken:
            return None

        sids = np.array([t[0] for t in taken])
        slots = np.array([t[1] for t in taken])
        rows = None if self.mask_rows is None else [self.mask_rows[self.sids.index(s)] for s in sids]
        pred = self.predictor.run(len(taken), rows, want_density=True)
        ring.density[sids, slots] = pred.density

        out = empty_batch(len(taken))
        out["sid"], out["slot"] = sids, slots
        out["frame_idx"] = [t[2] for t in taken]
        out["captured"] = ring.meta["timestamp"][sids, slots]
        out["count"] = pred.counts
        out["cx"], out["cy"] = pred.centroids[:, 0], pred.centroids[:, 1]
        out["inferred"] = time.time()
        return out


class InferencerProcess(mp.Process):
    def __init__(
        self,
        layout: RingLayout,
        results: mp.Queue,
        *,
        worker_id: int,
        num_workers: int,
        checkpoint: Path,
        device: str,
        mask_dir: Path | None,
        macs: Sequence[str | None],
        log_level: str = "INFO",
    ):
        super().__init__(name=f"Inferencer-{worker_id}", daemon=True)
        self.layout = layout
        self.results = results
        self.sids = [s for s in range(layout.num_streams) if s % num_workers == worker_id]
        self.checkpoint = checkpoint
        self.device = device
        self.mask_dir = mask_dir
        self.macs = list(macs)
        self.log_level = log_level
        self.ready = mp.Event()
        self.stop_event = mp.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        setup_logging(self.log_level)
        import torch

        from birdcount.core.inference import DensityPredictor
        from birdcount.core.masks import RegionMasks
        from birdcount.core.model import load_inference_model

        # Autotuning would make startup ~4 s slower and stall the first batch of
        # every new size, for ~5 % at batch 20.
        torch.backends.cudnn.benchmark = False
        ring = FrameRing.attach(self.layout)
        try:
            masks = RegionMasks.from_dir(
                self.mask_dir, [self.macs[s] for s in self.sids], self.layout.frame_size, name=self.name
            )
            model = load_inference_model(self.checkpoint, self.device, input_order="bgr")
            predictor = DensityPredictor(
                model, frame_size=self.layout.frame_size, masks=masks, max_batch=len(self.sids)
            )
            for n in sorted({1, len(self.sids)}):
                predictor.run(n, None if masks is None else [0] * n)
            loop = InferenceLoop(
                ring, predictor, self.sids, mask_rows=None if masks is None else range(len(self.sids))
            )
            logger.info("[%s] Ready for cameras %s.", self.name, self.sids)
            self.ready.set()
            self._serve(ring, loop)
        finally:
            ring.close()

    def _serve(self, ring: FrameRing, loop: InferenceLoop) -> None:
        while not self.stop_event.is_set():
            tick = ring.next_tick(time.monotonic())
            if self.stop_event.wait(max(0.0, tick - time.monotonic())):
                break
            try:
                batch = loop.step(tick + loop.max_wait)
            except Exception:
                logger.exception("[%s] Batch failed.", self.name)
                continue
            if batch is None:
                continue
            try:
                self.results.put(batch, timeout=_QUEUE_TIMEOUT)
            except queue.Full:
                ring.release(batch["sid"], batch["slot"])
                logger.warning("[%s] Result queue full; dropped %d frames.", self.name, len(batch))
