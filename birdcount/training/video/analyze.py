"""One pass over one video: decode, count, draw — three stages running at once.

    decoder thread  ->  GPU batches (this thread)  ->  drawing / encoding threads
    (core.video)         DensityPredictor               render.py, encode.py

Bounded queues between the stages keep memory flat however long the video.
Counting goes through `core.inference.DensityPredictor` like the live runtime
— fitted frames, region mask, fp16 — so a count here is the count the alarm
would have seen for that frame.

The run folder:
    summary.json     status, the video, the settings, summary numbers, artifacts
                     (written at the start with status "running" and rewritten at the end)
    samples.jsonl    one line per sample, appended batch by batch while the run goes
    timeline.csv     the samples with clock times and smoothed series
    timeline.png     the figure
    peaks/           overlays of the busiest moments
    frames/          every sampled frame (--frames)
    <mode>.mp4       the rendered video, H.264 (--render)
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from birdcount.core.inference import DensityPredictor
from birdcount.core.model import OUTPUT_STRIDE
from birdcount.core.video import (
    SampledFrames,
    VideoInfo,
    find_frame,
    fingerprint,
    probe,
    sampling_step,
    start_from_name,
)
from birdcount.training.video.encode import H264Writer
from birdcount.training.video.render import Renderer, save_jpeg
from birdcount.training.video.timeline import (
    Sample,
    clock_of,
    pick_peaks,
    summarize,
    window_peak,
    write_csv,
)


DRAW_WORKERS = 4
PROGRESS_SECONDS = 5.0


@dataclass(frozen=True)
class Options:
    every: float = 1.0  # seconds between samples; 0 = every frame
    seconds: float = 0.0  # analyse only the first N seconds; 0 = all
    max_samples: int = 5000
    start: datetime | None = None  # wall clock of the first frame; default from the filename
    window: int = 16  # peak window side, in density cells
    thresholds: dict[str, float] = field(default_factory=dict)  # "count" / "peak" -> level
    report: str = "both"  # series in the figure and the peak choice: both / count / peak
    smooth: int = 5
    frames: str = "none"  # render.FRAME_STYLES
    render: str = "none"  # render.VIDEO_MODES
    width: int = 0
    peaks: int = 4
    peak_gap: float = 15.0
    chart: bool = True


def analyze(
    path: Path,
    predictor: DensityPredictor,
    out_dir: Path,
    opts: Options,
    *,
    about: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Analyse `path` into `out_dir`; returns what summary.json holds.

    `about` is merged into the summary's settings (checkpoint, mask source).
    Ctrl+C stops the pass; what was sampled so far is still written out.
    """
    t0 = time.perf_counter()
    info = probe(path)
    limit = round(opts.seconds * info.fps) if opts.seconds > 0 else 0
    if info.frames:
        limit = min(limit, info.frames) if limit else info.frames
    step = sampling_step(info, opts.every, limit, opts.max_samples)
    start = opts.start or start_from_name(path.name)
    keep = None if predictor.masks is None else predictor.masks.keep[0]
    renderer = Renderer(predictor.frame_size, keep=keep, thresholds=opts.thresholds, start=start, width=opts.width)

    out_dir.mkdir(parents=True, exist_ok=True)
    window_px = opts.window * OUTPUT_STRIDE
    record = {
        "status": "running",
        "video": {
            "path": str(path.resolve()),
            "name": path.name,
            "width": info.width,
            "height": info.height,
            "fps": round(info.fps, 3),
            "frames": info.frames,
        },
        "settings": {
            **(about or {}),
            "frame_size": list(predictor.frame_size),
            "masked": keep is not None,
            "step": step,
            "interval": round(step / info.fps, 4),
            "frames_limit": limit,
            "start_time": start.isoformat(sep=" ") if start else None,
            "window": opts.window,
            "window_px": window_px,
            "thresholds": renderer.thresholds,
            "report": opts.report,
            "smooth": opts.smooth,
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(out_dir / "summary.json", record)
    log(
        f"{path.name}: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.frames or '?'} frames; "
        f"one sample every {step / info.fps:.2f}s, counted at {predictor.frame_size[0]}x{predictor.frame_size[1]}, "
        f"{'masked' if keep is not None else 'UNMASKED'} -> {out_dir}"
    )

    samples: list[Sample] = []
    prints: list[np.ndarray] = []
    sink = _Sink(out_dir, renderer, opts, fps=info.fps / step)
    # A batch of decoded frames ahead is enough to keep the GPU fed.
    reader = SampledFrames(path, step=step, limit=limit, size=predictor.frame_size, prefetch=predictor.max_batch)
    mask_ids = None if keep is None else [0] * predictor.max_batch
    # (frame index, the frame itself only while something will draw it): the
    # model reads its own copy in `staging`.
    batch: list[tuple[int, np.ndarray | None]] = []
    status = "finished"

    def flush() -> None:
        n = len(batch)
        pred = predictor.run(n, None if mask_ids is None else mask_ids[:n], want_density=True)
        lines = []
        for j, (index, frame) in enumerate(batch):
            density = pred.density[j]
            peak, box = window_peak(density, opts.window, OUTPUT_STRIDE)
            s = Sample(
                len(samples), index, round(index / info.fps, 3), round(float(pred.counts[j]), 3), round(peak, 3), box
            )
            samples.append(s)
            lines.append(json.dumps(s.to_json()) + "\n")
            if frame is not None:
                sink.submit(frame, density, s)
        jsonl.write("".join(lines))
        jsonl.flush()
        batch.clear()

    next_report = time.perf_counter() + PROGRESS_SECONDS
    try:
        with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as jsonl:
            for index, frame in reader:
                predictor.staging[len(batch)] = frame
                prints.append(fingerprint(frame))
                batch.append((index, frame if sink.active else None))
                if len(batch) == predictor.max_batch:
                    flush()
                    if time.perf_counter() >= next_report:
                        next_report += PROGRESS_SECONDS
                        log(_progress(reader.decoded, limit, len(samples), time.perf_counter() - t0))
            if batch:
                flush()
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        reader.close()
        sink.close()
    if not samples:
        raise RuntimeError(f"no frames decoded from {path}")

    artifacts: dict = {"samples": "samples.jsonl", "csv": "timeline.csv"}
    write_csv(out_dir / "timeline.csv", samples, start, opts.smooth)
    if sink.frames_dir is not None:
        artifacts["frames"] = {"dir": "frames", "style": opts.frames, "count": len(samples)}
    if sink.video_path is not None:
        artifacts["video"] = sink.video_path.name
    if status == "finished":
        if opts.chart and len(samples) > 1:
            from birdcount.training.video.chart import render_chart

            render_chart(
                out_dir / "timeline.png",
                samples,
                title=path.name,
                report=opts.report,
                start=start,
                thresholds=renderer.thresholds,
                smooth=opts.smooth,
                window_px=window_px,
                interval=step / info.fps,
            )
            artifacts["chart"] = "timeline.png"
        artifacts["peaks"] = _save_peaks(path, info, predictor, renderer, samples, prints, out_dir, opts, start, log)

    record.update(
        status=status,
        summary=summarize(samples, start),
        artifacts=artifacts,
        seconds=round(time.perf_counter() - t0, 2),
    )
    _write_json(out_dir / "summary.json", record)
    return record


def _save_peaks(
    path: Path,
    info: VideoInfo,
    predictor: DensityPredictor,
    renderer: Renderer,
    samples: list[Sample],
    prints: list[np.ndarray],
    out_dir: Path,
    opts: Options,
    start: datetime | None,
    log: Callable[[str], None],
) -> list[dict]:
    """Overlays of the busiest moments ("busiest" by the reported series).

    The frame is decoded again (found by its fingerprint, see `core.video`) and
    counted again for its heatmap; the caption keeps the timeline's numbers.
    """
    key = "count" if opts.report == "count" else "peak"
    chosen = pick_peaks(samples, opts.peaks, opts.peak_gap, key)
    if not chosen:
        return []
    peaks_dir = out_dir / "peaks"
    peaks_dir.mkdir(exist_ok=True)
    mask_ids = None if predictor.masks is None else [0]
    saved = []
    for rank, s in enumerate(chosen, start=1):
        frame = find_frame(path, s.frame, prints[s.i], predictor.frame_size)
        if frame is None:
            log(f"  [warn] could not decode frame {s.frame} of {info.path.name} again; no peak picture for it")
            continue
        density = predictor.predict([frame], mask_ids, want_density=True).density[0]
        name = f"peak_{rank:02d}_{int(s.t):06d}s.jpg"
        save_jpeg(peaks_dir / name, renderer.peak_image(frame, density, s))
        saved.append({**s.to_json(), "clock": clock_of(start, s.t), "path": f"peaks/{name}"})
    return saved


def frame_name(sample: Sample) -> str:
    """A saved sample's file in frames/: recording order, then the second it shows."""
    return f"frame_{sample.i:05d}_{int(sample.t):06d}s.jpg"


class _Sink:
    """Drawing and encoding on worker threads, off the GPU loop.

    Saved frames are independent JPEGs; video frames must reach the encoder in
    order, so their futures queue up for one writer thread. Both paths block
    `submit` when full, which is what bounds memory.
    """

    def __init__(self, out_dir: Path, renderer: Renderer, opts: Options, *, fps: float):
        self.renderer, self.style, self.mode = renderer, opts.frames, opts.render
        self.frames_dir = out_dir / "frames" if opts.frames != "none" else None
        self.video_path = out_dir / f"{opts.render}.mp4" if opts.render != "none" else None
        self.active = self.frames_dir is not None or self.video_path is not None
        if not self.active:
            return
        if self.frames_dir is not None:
            self.frames_dir.mkdir(exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=DRAW_WORKERS, thread_name_prefix="VideoDraw")
        self._slots = threading.BoundedSemaphore(2 * DRAW_WORKERS)
        self._errors: list[BaseException] = []
        self._ordered: queue.Queue[Future | None] | None = None
        if self.video_path is not None:
            self._writer = H264Writer(self.video_path, fps, renderer.video_size(self.mode))
            self._ordered = queue.Queue(maxsize=2 * DRAW_WORKERS)
            self._writer_thread = threading.Thread(target=self._write_video, name="VideoWriter", daemon=True)
            self._writer_thread.start()

    def submit(self, frame: np.ndarray, density: np.ndarray, sample: Sample) -> None:
        if not self.active:
            return
        if self._errors:
            raise self._errors[0]
        density = density.copy()  # the predictor reuses its buffer on the next batch
        if self.frames_dir is not None:
            self._slots.acquire()
            future = self._pool.submit(self._save_frame, frame, density, sample)
            future.add_done_callback(self._done)
        if self._ordered is not None:
            self._ordered.put(self._pool.submit(self.renderer.video_frame, self.mode, frame, density, sample))

    def _save_frame(self, frame: np.ndarray, density: np.ndarray, sample: Sample) -> None:
        save_jpeg(self.frames_dir / frame_name(sample), self.renderer.frame_image(self.style, frame, density, sample))

    def _done(self, future: Future) -> None:
        self._slots.release()
        if future.exception() is not None:
            self._errors.append(future.exception())

    def _write_video(self) -> None:
        while (future := self._ordered.get()) is not None:
            try:
                self._writer.write(future.result())
            except BaseException as e:
                self._errors.append(e)

    def close(self) -> None:
        if not self.active:
            return
        if self._ordered is not None:
            self._ordered.put(None)
            self._writer_thread.join()
            self._writer.close()
        self._pool.shutdown(wait=True)
        if self._errors:
            raise self._errors[0]


def _progress(decoded: int, limit: int, samples: int, elapsed: float) -> str:
    done = f"{decoded}/{limit} frames ({decoded / limit:.0%})" if limit else f"{decoded} frames"
    return f"  {done}, {samples} samples, {decoded / max(elapsed, 1e-6):.0f} frames/s"


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
