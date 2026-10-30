"""Decoding a video file: the sampled frames, in order, on a background thread,
and the wall clock its name says it started at.

Frames come out fitted to the model geometry (`core.frames.fit_frame`), the
same as a live camera's. Frames between two samples are skipped with
`grab()`, which demuxes and decodes but never converts to BGR: at one sample
a second that reads a 25 fps clip ~4x faster than decoding every frame.

Going back for one frame afterwards (`find_frame`) cannot trust seeking:
`CAP_PROP_POS_FRAMES` on the farm's MKV recordings lands 1-3 frames past the
target and reports the target. So every sampled frame leaves a fingerprint,
and a frame is found again by seeking a little earlier and decoding forward
until the fingerprint matches.

numpy + cv2 only: the video tools (training.video) and the replay of
recordings (deploy.replay) both read clips through here.
"""

from __future__ import annotations

import math
import queue
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from birdcount.core.frames import fit_frame


# A fingerprint is this grid of pixels, sampled straight out of the fitted
# frame: decoding is deterministic, so the same frame gives the same pixels,
# and sensor noise alone makes neighbouring frames differ.
_PRINT_GRID = (18, 32)
# Mean absolute difference (0-255) below which two fingerprints are one frame.
_PRINT_MATCH = 1.0
# How far before a frame `find_frame` seeks; seeks land after the target.
_SEEK_BACK = 48
# Start stamps in recording names: the farm export's `<MAC>_YYYYMMDD_HHMMSS_<id>`,
# the AXIS recorder's `YYYYMMDD_HHMMSS_<id>` and the deploy recorder's `YYYYMMDD-HHMMSS`.
_NAME_STAMP = re.compile(r"(?<!\d)(20\d{6})[_-](\d{6})(?!\d)")


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int  # as the container reports it; 0 when unknown

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def probe(path: Path) -> VideoInfo:
    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if 0 < fps <= 240 else 25.0,
            frames=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
    finally:
        cap.release()


def start_from_name(name: str) -> datetime | None:
    """The recording's start (local time), when its filename carries one."""
    match = _NAME_STAMP.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def sampling_step(info: VideoInfo, every: float, limit: int, max_samples: int) -> int:
    """Frames between two samples: `every` seconds (0 = every frame), stretched
    so that at most `max_samples` samples are taken from the first `limit` frames."""
    step = max(1, round(every * info.fps))
    if max_samples > 0 and limit > 0 and limit / step > max_samples:
        step = math.ceil(limit / max_samples)
    return step


def fingerprint(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    rows = np.linspace(0, h - 1, _PRINT_GRID[0]).astype(np.intp)
    cols = np.linspace(0, w - 1, _PRINT_GRID[1]).astype(np.intp)
    return frame[np.ix_(rows, cols)]


def _print_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b).mean())


class SampledFrames:
    """Iterate (frame index, fitted BGR frame) for every `step`-th frame of the
    first `limit` (0 = all), decoded ahead on a thread into a bounded queue."""

    _DONE = object()

    def __init__(self, path: Path, *, step: int, limit: int, size: tuple[int, int], prefetch: int = 32):
        self.path, self.step, self.limit, self.size = path, step, limit, size
        self.decoded = 0  # frames demuxed so far, sampled or not
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="VideoDecoder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cap = None
        try:
            cap = _open(self.path)
            index = 0
            while not self._stop.is_set() and (self.limit <= 0 or index < self.limit):
                if index % self.step:
                    ok = cap.grab()
                    item = None
                else:
                    ok, frame = cap.read()
                    item = (index, fit_frame(frame, self.size)) if ok else None
                if not ok:
                    break
                index += 1
                self.decoded = index
                if item is not None:
                    self._put(item)
        except BaseException as e:  # re-raised on the consumer's thread
            self._error = e
        finally:
            if cap is not None:
                cap.release()
            self._put(self._DONE)

    def _put(self, item) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        while True:
            try:
                item = self._queue.get(timeout=0.2)  # a timed wait, so Ctrl+C gets through on Windows
            except queue.Empty:
                continue
            if item is self._DONE:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        self._thread.join()


def find_frame(path: Path, index: int, wanted: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """The fitted frame `index` whose fingerprint is `wanted`, or None.

    Seeks `_SEEK_BACK` frames early and decodes forward; if the frame is not
    in that window, decodes from the start (exact, slower).
    """
    start = max(0, index - _SEEK_BACK)
    for seek in (True, False) if start else (False,):
        cap = _open(path)
        try:
            if seek:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            elif not all(cap.grab() for _ in range(start)):
                return None
            found = _scan(cap, wanted, size, 2 * _SEEK_BACK + 1)
        finally:
            cap.release()
        if found is not None:
            return found
    return None


def _scan(cap: cv2.VideoCapture, wanted: np.ndarray, size: tuple[int, int], frames: int) -> np.ndarray | None:
    best, best_distance = None, _PRINT_MATCH
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = fit_frame(frame, size)
        distance = _print_distance(fingerprint(frame), wanted)
        if distance < best_distance:
            best, best_distance = frame, distance
            if distance == 0:
                break
    return best


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open video {path}")
    return cap
