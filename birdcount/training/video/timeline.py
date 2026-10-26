"""What one video's samples say: crowding over time, its summary, its busiest moments.

Two numbers per sampled frame, both straight from the density map and so in
birds, comparable across clips and hours:

  count  the whole (masked) map integrated: every bird the camera counts —
         what the deployed alarm compares with its threshold
  peak   the busiest `window` x `window` patch of the map integrated: birds
         standing inside one small area. This is the pile-up signal; the
         frame count can stay flat while birds bunch into a corner.

numpy + cv2 only.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np


# Decimals per series wherever a number is shown: a bird count is whole, while
# the peak is a small integral whose first decimal carries information.
DECIMALS = {"count": 0, "peak": 1}

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%H:%M:%S", "%H:%M")
# Start stamps in recording names: the farm export's `<MAC>_YYYYMMDD_HHMMSS_<id>`
# and the deploy recorder's `YYYYMMDD-HHMMSS`.
_NAME_STAMP = re.compile(r"(?<!\d)(20\d{6})[_-](\d{6})(?!\d)")


@dataclass(frozen=True)
class Sample:
    i: int  # sample number
    frame: int  # frame index in the video
    t: float  # seconds from the start of the video
    count: float
    peak: float
    box: tuple[int, int, int, int]  # the peak window, (x0, y0, x1, y1) in frame pixels

    def to_json(self) -> dict:
        return asdict(self)


def window_peak(density: np.ndarray, window: int, stride: int) -> tuple[float, tuple[int, int, int, int]]:
    """Birds inside the busiest `window` x `window`-cell patch of `density`, and that
    patch as a box in frame pixels (one cell = `stride` pixels).

    A window larger than the map shrinks to it along that side, so the peak of
    a map smaller than the window is its total.
    """
    h, w = density.shape
    sh, sw = max(1, min(window, h)), max(1, min(window, w))
    ii = cv2.integral(np.asarray(density, np.float32), sdepth=cv2.CV_64F)
    sums = ii[sh:, sw:] - ii[:-sh, sw:] - ii[sh:, :-sw] + ii[:-sh, :-sw]
    row, col = (int(v) for v in np.unravel_index(int(np.argmax(sums)), sums.shape))
    box = (col * stride, row * stride, (col + sw) * stride, (row + sh) * stride)
    return float(sums[row, col]), box


def pick_peaks(samples: list[Sample], n: int, gap: float, key: str) -> list[Sample]:
    """The `n` highest samples by `key`, at least `gap` seconds apart (so not
    `n` pictures of one event), in time order."""
    chosen: list[Sample] = []
    for s in sorted(samples, key=lambda s: getattr(s, key), reverse=True):
        if len(chosen) >= n:
            break
        if all(abs(s.t - c.t) >= gap for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda s: s.t)


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Centred rolling mean, edge-padded so the line spans the whole axis."""
    if window <= 1 or not values:
        return list(values)
    half = window // 2
    padded = np.pad(np.asarray(values, np.float64), half, mode="edge")
    return np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")[: len(values)].tolist()


def parse_start_time(text: str) -> datetime:
    """'16:20' / '16:20:05' / '2026-08-28 16:20:00' -> datetime."""
    for fmt in _TIME_FORMATS:
        try:
            stamp = datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
        return stamp if "%Y" in fmt else stamp.replace(year=2000, month=1, day=1)
    raise ValueError(f"start time expects HH:MM, HH:MM:SS or 'YYYY-MM-DD HH:MM:SS' (got {text!r})")


def start_from_name(name: str) -> datetime | None:
    """The recording's start, when its filename carries one."""
    match = _NAME_STAMP.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def clock_of(start: datetime | None, seconds: float) -> str:
    """Wall-clock time of a moment, or elapsed h:mm:ss when the start is unknown."""
    if start is not None:
        return (start + timedelta(seconds=seconds)).strftime("%H:%M:%S")
    total = round(seconds)
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def summarize(samples: list[Sample], start: datetime | None) -> dict:
    out: dict = {"samples": len(samples)}
    for key in ("count", "peak"):
        values = np.array([getattr(s, key) for s in samples], np.float64)
        top = samples[int(values.argmax())]
        out[key] = {
            "mean": round(float(values.mean()), 2),
            "min": round(float(values.min()), 2),
            "p95": round(float(np.percentile(values, 95)), 2),
            "max": round(float(values.max()), 2),
            "max_t": top.t,
            "max_clock": clock_of(start, top.t),
        }
    return out


def write_csv(path: Path, samples: list[Sample], start: datetime | None, smooth: int) -> None:
    counts = rolling_mean([s.count for s in samples], smooth)
    peaks = rolling_mean([s.peak for s in samples], smooth)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "frame", "seconds", "clock", "count", "peak", "count_smooth", "peak_smooth"])
        for s, c, p in zip(samples, counts, peaks, strict=True):
            writer.writerow(
                [
                    s.i,
                    s.frame,
                    f"{s.t:.3f}",
                    clock_of(start, s.t),
                    f"{s.count:.3f}",
                    f"{s.peak:.4f}",
                    f"{c:.3f}",
                    f"{p:.4f}",
                ]
            )
