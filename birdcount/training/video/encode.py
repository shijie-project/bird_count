"""Rendered videos as H.264 MP4, the one codec every browser plays.

OpenCV's writer gives mp4v (MPEG-4 Part 2), which browsers refuse, and its
H.264 path depends on an OpenH264 DLL that is rarely installed. PyAV's
wheels carry FFmpeg with libx264 and NVENC, so H.264 comes out on every
machine: on the GPU's video engine when there is one (faster than mp4v and
a quarter of its size), else with libx264. `+faststart` puts the index
first, so a browser starts playing before the whole file has loaded.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


# Encoder and options, best first. Both at least as fast as OpenCV's mp4v on
# 960 px overlays (~520 / ~440 vs ~410 frames/s) at ~20 / ~40 MB per 5-minute
# clip at 5 fps (mp4v: ~85 MB).
ENCODERS = (
    ("h264_nvenc", {"preset": "p5", "cq": "26"}),
    ("libx264", {"preset": "superfast", "crf": "26"}),
)


@functools.cache
def encoder(width: int, height: int) -> tuple[str, dict[str, str]]:
    """The first of ENCODERS this machine can open at this size (NVENC refuses tiny frames)."""
    import av

    for name, options in ENCODERS[:-1]:
        try:
            ctx = av.CodecContext.create(name, "w")
            ctx.width, ctx.height, ctx.pix_fmt, ctx.time_base = width, height, "yuv420p", Fraction(1, 5)
            ctx.options = dict(options)
            ctx.open()
            return name, options
        except (av.FFmpegError, ValueError):  # not built in, or no such hardware
            continue
    return ENCODERS[-1]


class H264Writer:
    """Write BGR frames of one size; odd dimensions lose their last row or column (4:2:0 needs even ones)."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        import av  # FFmpeg's DLLs load only when a video is actually rendered

        self.width, self.height = size[0] - size[0] % 2, size[1] - size[1] % 2
        if self.width < 2 or self.height < 2:
            raise ValueError(f"frame size {size} is too small for a video")
        self.codec, options = encoder(self.width, self.height)
        self._container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self._stream = self._container.add_stream(self.codec, rate=Fraction(fps).limit_denominator(1000))
        self._stream.width, self._stream.height = self.width, self.height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = dict(options)
        self._frame_type = av.VideoFrame

    def write(self, frame: np.ndarray) -> None:
        # OpenCV's conversion (BT.601, as FFmpeg's) runs without the GIL; FFmpeg's
        # own would hold it and slow the drawing threads.
        yuv = cv2.cvtColor(frame[: self.height, : self.width], cv2.COLOR_BGR2YUV_I420)
        for packet in self._stream.encode(self._frame_type.from_ndarray(yuv, format="yuv420p")):
            self._container.mux(packet)

    def close(self) -> None:
        try:
            for packet in self._stream.encode():  # flush the encoder's delayed frames
                self._container.mux(packet)
        finally:
            self._container.close()
