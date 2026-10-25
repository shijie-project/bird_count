"""Opening a camera stream or a video file for decoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2


logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mkv", ".mp4", ".avi", ".mov", ".ts")

# FFmpeg's default frame threading starts ~40 threads per decoder. With one
# reader thread per camera the cameras already decode in parallel, and for 20
# cameras the default cost +1.4 GB RSS and 1388 threads for *more* CPU per
# frame; one thread each measured lowest on both.
_DECODER_THREADS = 1


@dataclass(frozen=True)
class Source:
    """One camera: a stream URL, or a video file replayed as if it were live."""

    sid: int
    uri: str
    is_file: bool

    @property
    def label(self) -> str:
        return f"cam{self.sid}"


@dataclass
class Capture:
    cap: cv2.VideoCapture
    hw_decode: bool
    fps: float | None  # the file's own frame rate, when it has a usable one


def open_capture(source: Source, *, hw_decode: bool) -> Capture | None:
    """Open with FFmpeg, on the GPU video engine when asked and available.

    A stream the hardware decoder cannot handle is reopened in software, so
    `hw_decode` is a preference, never a reason to lose a camera.
    """
    attempts = [True, False] if hw_decode else [False]
    for hw in attempts:
        params = [cv2.CAP_PROP_N_THREADS, _DECODER_THREADS]
        if hw:
            params += [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        cap = cv2.VideoCapture(source.uri, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            cap.release()
            continue
        active = hw and cap.get(cv2.CAP_PROP_HW_ACCELERATION) > 0
        fps = cap.get(cv2.CAP_PROP_FPS) if source.is_file else 0.0
        return Capture(cap=cap, hw_decode=active, fps=fps if fps and 0 < fps <= 240 else None)
    return None
