"""Pictures of a video's samples: overlay frames, peak frames, the rendered video.

Heatmap and region mask come from `core.visualize`, so a picture reads
exactly like the live monitor: same absolute colour scale, ignored area
tinted, the counted region outlined. A caption carries the time and the
numbers, red when one is over its threshold.

Everything is composed at the analysed size (the model geometry) and scaled
to the output width last, with the caption drawn after scaling so it stays
legible in a thumbnail.

numpy + cv2 only.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from birdcount.core.visualize import (
    Scratch,
    density_to_heatmap,
    draw_text,
    mask_layers,
    overlay_density,
    overlay_mask,
)
from birdcount.training.video.timeline import DECIMALS, Sample, clock_of


ALERT = (60, 60, 255)  # BGR red: a number over its threshold
PEAK_BOX = (255, 255, 255)
JPEG_QUALITY = 92

# --frames styles and --render modes.
FRAME_STYLES = ("none", "overlay", "boxes", "plain")
VIDEO_MODES = ("none", "overlay", "split", "density")


class Renderer:
    def __init__(
        self,
        size: tuple[int, int],
        *,
        keep: np.ndarray | None = None,
        thresholds: dict[str, float] | None = None,
        start: datetime | None = None,
        width: int = 0,
    ):
        self.size = size  # (width, height) of the analysed frames
        self.layers = None if keep is None else mask_layers(keep, size)
        self.thresholds = {k: v for k, v in (thresholds or {}).items() if v > 0}
        self.start = start
        self.width = width  # of the output images; 0 = the analysed size
        self._local = threading.local()  # one Scratch per drawing thread

    # -- the pieces --------------------------------------------------------

    def caption(self, sample: Sample, keys: tuple[str, ...]) -> tuple[str, bool]:
        """'16:21:05  count 173 / thr 150 OVER  peak 7.2', and whether anything is over."""
        parts, over = [clock_of(self.start, sample.t)], False
        for key in keys:
            value, decimals = getattr(sample, key), DECIMALS[key]
            text = f"{key} {value:.{decimals}f}"
            threshold = self.thresholds.get(key)
            if threshold is not None:
                text += f" / thr {threshold:.{decimals}f}"
                if value >= threshold:
                    text += " OVER"
                    over = True
            parts.append(text)
        return "  ".join(parts), over

    def overlay(self, frame: np.ndarray, density: np.ndarray, sample: Sample | None = None) -> np.ndarray:
        """Frame + region mask + heatmap, and the peak box when `sample` is given."""
        scratch = getattr(self._local, "scratch", None)
        if scratch is None:
            scratch = self._local.scratch = Scratch()
        out = frame.copy()
        if self.layers is not None:
            overlay_mask(out, self.layers, scratch=scratch)
        overlay_density(out, density, scratch=scratch)
        if sample is not None:
            self._peak_box(out, sample)
        return out

    def heatmap(self, density: np.ndarray) -> np.ndarray:
        """The density map alone, full colour scale, at the analysed size."""
        up = cv2.resize(np.asarray(density, np.float32), self.size, interpolation=cv2.INTER_LINEAR)
        return density_to_heatmap(up)[0]

    def finish(
        self, image: np.ndarray, caption: str = "", over: bool = False, *, width: int | None = None
    ) -> np.ndarray:
        """Scale to the output width, then caption on a darkened band (legible over
        the mask tint and any heatmap colour). Unscaled, draws on `image` itself."""
        width = self.width if width is None else width
        if width and image.shape[1] != width:
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if caption:
            scale = max(0.6, image.shape[1] / 1280)
            margin = max(8, int(12 * scale))
            (text_w, text_h), base = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, round(scale * 2))
            )
            band = image[: text_h + base + 2 * margin, : text_w + 2 * margin]
            np.multiply(band, 0.35, out=band, casting="unsafe")
            draw_text(image, caption, (margin, text_h + margin), scale, ALERT if over else (255, 255, 255))
        return image

    # -- what the tool saves -----------------------------------------------

    def frame_image(self, style: str, frame: np.ndarray, density: np.ndarray, sample: Sample) -> np.ndarray:
        """One sampled frame for frames/: 'plain' untouched (for annotating), 'overlay'
        with the count, 'boxes' adding the peak window and its number."""
        if style == "plain":
            return self.finish(frame)
        boxes = style == "boxes"
        image = self.overlay(frame, density, sample if boxes else None)
        return self.finish(image, *self.caption(sample, ("count", "peak") if boxes else ("count",)))

    def peak_image(self, frame: np.ndarray, density: np.ndarray, sample: Sample) -> np.ndarray:
        """A busiest moment, full size whatever the output width: these go in reports."""
        image = self.overlay(frame, density, sample)
        return self.finish(image, *self.caption(sample, ("peak", "count")), width=0)

    def video_frame(self, mode: str, frame: np.ndarray, density: np.ndarray, sample: Sample) -> np.ndarray:
        caption = self.caption(sample, ("count",))
        if mode == "overlay":
            return self.finish(self.overlay(frame, density), *caption)
        if mode == "split":
            return self.finish(np.hstack([self.overlay(frame, density), self.heatmap(density)]), *caption)
        if mode == "density":
            return self.finish(self.heatmap(density), *caption)
        raise ValueError(f"unknown video mode {mode!r}")

    def video_size(self, mode: str) -> tuple[int, int]:
        w, h = self.size
        if mode == "split":
            w *= 2
        if self.width and self.width != w:
            return self.width, max(1, round(h * self.width / w))
        return w, h

    def _peak_box(self, image: np.ndarray, sample: Sample) -> None:
        x0, y0, x1, y1 = sample.box
        cv2.rectangle(image, (x0, y0), (x1, y1), PEAK_BOX, max(2, round(image.shape[1] / 800)))
        scale = max(0.7, image.shape[1] / 1100)
        text = f"peak {sample.peak:.{DECIMALS['peak']}f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        pad = max(4, int(6 * scale))
        x = min(max(x0 + pad, pad), max(pad, image.shape[1] - text_w - pad))
        y = y0 - pad if y0 - pad - text_h > 0 else y0 + text_h + pad  # above the box, else inside it
        draw_text(image, text, (x, y), scale, PEAK_BOX)


def save_jpeg(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError(f"cannot encode {path.name}")
    buf.tofile(path)
