"""One camera's day of recordings through the deployed pipeline, clip after clip.

Every sampled frame goes the way a live camera's does: counted by
`DensityPredictor` with the camera's region mask, decided on by the
`AlarmEngine` (motion filter, smoothing, hysteresis, Level 1/2/3), logged by
the `CountLogger` (a row per second, snapshots per count bin), and on an
alarm action written as evidence by the SMS handler's dispatcher, with
nothing sent. Only the clock differs: a sample's time is its clip's start,
from the file name, plus the frame's offset in the clip, and frames are read
as fast as they decode instead of at the camera's pace.

Frames are sampled at the runtime's FPS (every 5th frame of a 25 fps clip at
FPS=5): the rate the alarm's timings and smoothing were set for.

Besides the count log and the evidence, every sample is kept as counted
(samples/<MAC>/<day>.npz, SAMPLE_FIELDS): enough to replay the alarm with
other thresholds, or to pick the clips worth counting again, without decoding
anything.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from birdcount.core.video import SampledFrames, probe
from birdcount.deploy.alarm import AppConfig, EvidenceStore
from birdcount.deploy.audit import NULL_AUDIT
from birdcount.deploy.handlers.base import Batch
from birdcount.deploy.handlers.count_logger import CountLogger
from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.handlers.sms import AlarmDispatcher, AlarmJob
from birdcount.deploy.replay.cache import InPlace, Stopped
from birdcount.deploy.replay.recordings import Clip
from birdcount.deploy.results import empty_batch


if TYPE_CHECKING:
    from birdcount.core.inference import DensityPredictor


logger = logging.getLogger(__name__)

ALL_DAY = "00:00-23:59"  # the count log's window: every recorded second
PROGRESS_SECONDS = 60.0
# Shares of the threshold the summary counts seconds above: the snapshot floor, `high`, the threshold.
OVER = (0.6, 0.8, 1.0)
SAMPLE_FIELDS = {
    "t": np.float64,  # unix time
    "count": np.float32,
    "cx": np.float32,  # density centroid, 0..1 of the frame; NaN when empty
    "cy": np.float32,
    "level": np.int8,  # alarm level of the camera's open event after this sample; 0 none
    "excluded": np.bool_,  # the motion filter set this sample aside
    "clip": np.int16,  # index into the day's clips
    "frame": np.int32,  # frame index in that clip
}


@dataclass(frozen=True)
class DayJob:
    mac: str
    day: str  # YYYY-MM-DD
    clips: tuple[Clip, ...]


@dataclass(frozen=True)
class Outputs:
    count_log: Path  # the count log's root (the day's folder goes under it)
    evidence: Path  # this camera-day's evidence folder
    samples: Path  # this camera-day's .npz


class ClipSource(Protocol):
    """Where a camera-day's clip `i` is read from: `cache.InPlace` or a `cache.ClipCache` plan."""

    def fetch(self, i: int, stop: threading.Event | None = None) -> Path: ...

    def done(self, i: int) -> None: ...


class _Frames:
    """Stands in for the frame ring in a `Batch`: `frames[sid, slot]` is the batch's frame `slot`."""

    __slots__ = ("_frames",)

    def __init__(self, frames: Sequence[np.ndarray]):
        self._frames = frames

    @property
    def frames(self) -> _Frames:
        return self

    def __getitem__(self, key: tuple[int, int]) -> np.ndarray:
        return self._frames[key[1]]


class _Day:
    """The pipeline of one camera-day and what it has counted so far."""

    def __init__(self, job: DayJob, predictor: DensityPredictor, config: AppConfig, out: Outputs, logger_kw: dict):
        self.job = job
        self.predictor = predictor
        self.engine = AlarmEngine(config, [job.mac])
        self.counts = CountLogger(out.count_log, ALL_DAY, self.engine, [job.mac], **logger_kw)
        self.evidence = AlarmDispatcher(
            EvidenceStore(out.evidence, config.rules), None, out.evidence / "snapshots", NULL_AUDIT
        )
        self.mask_ids = None if predictor.masks is None else [0] * predictor.max_batch
        self.columns: dict[str, list[np.ndarray]] = {k: [] for k in SAMPLE_FIELDS}
        self.alarms: dict[str, int] = {}
        self.last_t = -math.inf
        self.samples = 0

    def run(self, n: int, frames: list[np.ndarray], times: list[float], index: list[int], clip: int) -> None:
        """Count, decide on, log and file one batch, staged in the predictor's `staging[:n]`."""
        pred = self.predictor.run(n, None if self.mask_ids is None else self.mask_ids[:n])
        rec = empty_batch(n)
        rec["slot"] = np.arange(n)
        rec["frame_idx"] = index
        rec["captured"] = rec["inferred"] = times
        rec["count"] = pred.counts
        rec["cx"], rec["cy"] = pred.centroids[:, 0], pred.centroids[:, 1]
        decisions, _ = self.engine.evaluate(rec, set())
        self.counts.handle(Batch(rec, decisions, frozenset(), _Frames(frames)))
        levels = np.zeros(n, np.int8)
        excluded = np.zeros(n, np.bool_)
        for i, d in enumerate(decisions):
            if d is None:
                continue
            levels[i], excluded[i] = d.level, d.sample.motion_excluded
            self.evidence.process(
                AlarmJob(
                    sample=d.sample, event_id=d.event_id, actions=d.actions, frame=frames[i] if d.actions else None
                )
            )
            for a in d.actions:
                self.alarms[a.action_type] = self.alarms.get(a.action_type, 0) + 1
        for k, v in (
            ("t", rec["captured"]),
            ("count", rec["count"]),
            ("cx", rec["cx"]),
            ("cy", rec["cy"]),
            ("level", levels),
            ("excluded", excluded),
            ("clip", np.full(n, clip)),
            ("frame", rec["frame_idx"]),
        ):
            self.columns[k].append(v.astype(SAMPLE_FIELDS[k], copy=False))
        self.last_t = times[-1]
        self.samples += n

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v) if v else np.empty(0, SAMPLE_FIELDS[k]) for k, v in self.columns.items()}


def replay_day(
    job: DayJob,
    predictor: DensityPredictor,
    config: AppConfig,
    out: Outputs,
    *,
    fps: float,
    meta: dict,
    model_path: Path | None,
    stop: threading.Event,
    source: ClipSource | None = None,
) -> dict | None:
    """Count `job`'s clips in order into the count log, the evidence and the samples file.

    `source` hands over each clip's file to decode (a local copy, see
    `cache`); by default the clips are read where they are.

    Returns the camera-day's record (its numbers and one entry per clip), or
    None when `stop` was set first. A clip that cannot be read is recorded
    with its error and skipped; samples that would go back in time (a clip
    overlapping the one before) are dropped, as the alarm's clock only runs forward.
    """
    source = source or InPlace([c.path for c in job.clips])
    t0 = time.perf_counter()
    day = _Day(job, predictor, config, out, {"meta": meta, "model_path": model_path})
    clips: list[dict] = []
    next_report = t0 + PROGRESS_SECONDS
    day.counts.start()
    try:
        for ci, clip in enumerate(job.clips):
            if stop.is_set():
                return None
            try:
                path = source.fetch(ci, stop)
            except Stopped:
                return None
            except OSError as e:
                logger.warning("[%s %s] Skipping %s: %s", job.mac, job.day, clip.path.name, e)
                clips.append(_entry(clip, error=str(e)))
                continue
            try:
                clips.append(_count_clip(day, ci, clip, path, fps, stop))
            finally:
                source.done(ci)
            if time.perf_counter() >= next_report:
                next_report += PROGRESS_SECONDS
                elapsed = time.perf_counter() - t0
                logger.info(
                    "[%s %s] clip %d/%d (%s), %d samples, %.0f samples/s",
                    job.mac,
                    job.day,
                    ci + 1,
                    len(job.clips),
                    f"{clip.start:%H:%M}",
                    day.samples,
                    day.samples / elapsed,
                )
    finally:
        day.counts.stop()
    if stop.is_set():
        return None

    arrays = day.arrays()
    out.samples.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.samples.with_name(f"{out.samples.stem}.tmp.npz")
    np.savez_compressed(tmp, **arrays, clips=np.array([str(c.path) for c in job.clips]))
    tmp.replace(out.samples)
    elapsed = time.perf_counter() - t0
    return _record(job, day, arrays, clips, fps, elapsed)


def _entry(clip: Clip, error: str | None = None) -> dict:
    return {"path": str(clip.path), "start": f"{clip.start:%H:%M:%S}", "samples": 0, "error": error, "dropped": 0}


def _count_clip(day: _Day, ci: int, clip: Clip, path: Path, fps: float, stop: threading.Event) -> dict:
    """Count one clip, read from `path` (the clip itself or its local copy)."""
    entry = _entry(clip)
    try:
        info = probe(path)
    except OSError as e:
        logger.warning("[%s %s] Skipping %s: %s", day.job.mac, day.job.day, clip.path.name, e)
        entry["error"] = str(e)
        return entry
    predictor = day.predictor
    step = max(1, round(info.fps / fps))
    start = clip.start.timestamp()
    first, chunk = day.samples, len(day.columns["count"])
    frames: list[np.ndarray] = []
    times: list[float] = []
    index: list[int] = []

    def flush() -> None:
        day.run(len(frames), frames, times, index, ci)
        frames.clear()
        times.clear()
        index.clear()

    reader = SampledFrames(path, step=step, limit=0, size=predictor.frame_size, prefetch=predictor.max_batch)
    try:
        for i, frame in reader:
            if stop.is_set():
                break
            t = start + i / info.fps
            if t <= day.last_t:
                entry["dropped"] += 1
                continue
            predictor.staging[len(frames)] = frame
            frames.append(frame)
            times.append(t)
            index.append(i)
            if len(frames) == predictor.max_batch:
                flush()
    except Exception as e:  # a corrupt or truncated clip: keep what decoded
        logger.warning(
            "[%s %s] %s stopped decoding at frame %d: %s", day.job.mac, day.job.day, clip.path.name, reader.decoded, e
        )
        entry["error"] = str(e)
    finally:
        reader.close()
    if frames:
        flush()

    n = day.samples - first
    entry["samples"] = n
    if n:
        counts = np.concatenate(day.columns["count"][chunk:])
        entry.update(
            end=time.strftime("%H:%M:%S", time.localtime(day.last_t)),
            count_mean=round(float(counts.mean()), 1),
            count_p95=round(float(np.percentile(counts, 95)), 1),
            count_max=round(float(counts.max()), 1),
            max_level=int(np.concatenate(day.columns["level"][chunk:]).max()),
        )
    return entry


def _record(job: DayJob, day: _Day, arrays: dict, clips: list[dict], fps: float, elapsed: float) -> dict:
    counts = arrays["count"]
    threshold = day.engine.threshold_of(0)
    seconds = 1.0 / fps
    over = {}
    if threshold is not None and len(counts):
        over = {str(r): round(float((counts >= r * threshold).sum()) * seconds) for r in OVER}
    stats = {"mean": None, "p50": None, "p95": None, "max": None}
    if len(counts):
        stats = {
            "mean": round(float(counts.mean()), 1),
            "p50": round(float(np.percentile(counts, 50)), 1),
            "p95": round(float(np.percentile(counts, 95)), 1),
            "max": round(float(counts.max()), 1),
        }
    return {
        "mac": job.mac,
        "day": job.day,
        "camera_id": day.engine.camera_of.get(0),
        "threshold": threshold,
        "fps": fps,
        "samples": len(counts),
        "hours": len(counts) * seconds / 3600,
        "count": stats,
        "seconds_over": over,
        "alarms": day.alarms,
        "counted_in_s": round(elapsed, 1),
        "samples_per_s": round(len(counts) / max(elapsed, 1e-6), 1),
        "clips": clips,
    }
