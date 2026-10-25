"""Evaluation figures. All heatmaps come from `core.visualize`, so an evaluation
picture reads exactly like the live monitor."""

from __future__ import annotations

import cv2
import numpy as np

from birdcount.core.visualize import overlay_density
from birdcount.training.eval.metrics import within_tolerance
from birdcount.training.eval.regions import Region


OK = (45, 170, 45)  # BGR
OVER = (35, 35, 220)
UNDER = (220, 90, 25)
WHITE = (255, 255, 255)
HEADER = (0, 255, 255)

# Per-blob labels on the count overlay: blobs integrating at least this many
# birds (or holding a GT point) get "pred/gt"; red when off by more than
# max(1 bird, rel_tol * gt).
BLOB_LABEL_MIN_COUNT = 0.3
BLOB_ERR_ABS_FLOOR = 1.0


def draw_text(image: np.ndarray, text: str, org: tuple[int, int], scale: float, color=WHITE) -> None:
    """Text with a black outline, legible over any heatmap colour."""
    thick = max(1, round(scale * 2))
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _error_color(error: float, good_error: float) -> tuple[int, int, int]:
    return OK if abs(error) <= good_error else OVER if error > 0 else UNDER


def _header(image: np.ndarray, lines: list[str], color, scale: float) -> None:
    h = image.shape[0]
    x0, y0, step = int(h * 0.04), int(h * 0.08), int(scale * 34)
    for i, line in enumerate(lines):
        draw_text(image, line, (x0, y0 + i * step), scale, color)


def count_overlay(
    frame: np.ndarray, density: np.ndarray, blobs: list[Region], *, gt: float, pred: float
) -> np.ndarray:
    """Heatmap + per-blob "pred/gt" + the image's GT, prediction and error."""
    out = overlay_density(frame.copy(), density)
    scale = max(0.5, min(2.5, out.shape[0] / 600.0))
    for blob in blobs:
        if blob.count < BLOB_LABEL_MIN_COUNT and not blob.gt:
            continue
        bad = abs(blob.error) > max(BLOB_ERR_ABS_FLOOR, 0.10 * blob.gt)
        x, y = blob.centroid
        draw_text(out, f"{blob.count:.1f}/{blob.gt}", (int(x), int(y)), scale * 0.7, OVER if bad else WHITE)
    error = pred - gt
    rel = f"{error / gt * 100:+.1f}%" if gt > 0 else "n/a"
    color = (0, 180, 0) if within_tolerance(error, gt) else (0, 0, 255)
    _header(out, [f"GT: {gt:.1f}  Pred: {pred:.1f}", f"Abs Err: {abs(error):.1f}  Rel Err: {rel}"], color, scale)
    return out


def region_overlay(
    frame: np.ndarray,
    density: np.ndarray | None,
    labels: np.ndarray,
    regions: list[Region],
    *,
    good_error: float,
    label_min_error: float,
    header: list[str],
) -> np.ndarray:
    """Regions tinted by their error (green ok / red over / blue under), outlined,
    labelled with the signed error. `density` adds the heatmap underneath (blobs)."""
    out = frame.copy() if density is None else overlay_density(frame.copy(), density)
    h, w = out.shape[:2]
    full = cv2.resize(labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    tint = out.copy()
    for r in regions:
        tint[full == r.id] = _error_color(r.error, good_error)
    out = cv2.addWeighted(out, 0.80, tint, 0.20, 0)

    scale = max(0.35, min(1.0, h / 1100.0))
    for r in regions:
        color = _error_color(r.error, good_error)
        contours, _ = cv2.findContours((full == r.id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1 if r.row else 2)
        if abs(r.error) < label_min_error:
            continue
        if r.row:  # grid tile: "gt/pred (error)" in the corner
            x, y = int(r.bbox[0]) + 5, int(r.bbox[1]) + max(15, int(22 * scale))
            draw_text(out, f"{r.gt}/{r.count:.1f} ({r.error:+.1f})", (x, y), scale)
        else:
            draw_text(out, f"{r.error:+.1f}", (int(r.centroid[0]), int(r.centroid[1])), scale, color)
    _header(out, header, HEADER, max(0.5, min(1.4, h / 900.0)))
    return out
