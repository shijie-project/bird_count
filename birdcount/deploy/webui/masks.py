"""Editing a camera's region mask from the farm backend: outline, save, history.

A mask stays what the runtime, the evaluation and the training read: a
black/white PNG named by the camera's MAC in MASK_DIR (core.masks). Here it is
edited as polygons: the white area is traced into its outline, the page moves
the points, and saving fills the polygons back in at the file's own size.
Every save and restore first copies the file it replaces into
MASK_DIR/_history/<name>.<YYYYmmdd-HHMMSS>.png, so no mask is ever lost. A
save also keeps the picture the mask was drawn on, MASK_DIR/_reference/<name>.jpg
(core.masks.REFERENCE_DIR), for checking later that the camera has not moved
(deploy.replay.alignment).

numpy + cv2: the server imports this only when a mask is opened or saved.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from birdcount.core.frames import read_image
from birdcount.core.masks import BLACK_LEVEL, REFERENCE_DIR


HISTORY_DIR = "_history"
TRACE_TOLERANCE_PX = 1.5  # outline simplification: the file's own straight edges come back as single segments
MIN_AREA = 0.002  # a traced region smaller than this share of the frame is a speck, not a region
MAX_POINTS = 400


def _keep(path: Path) -> np.ndarray:
    img = read_image(path)
    return (~np.all(img < BLACK_LEVEL, axis=2)).astype(np.uint8)


def outline(path: Path) -> dict:
    """{size: [w, h], polygons: [[[x, y], ...]], keep: share counted} of a mask file."""
    keep = _keep(path)
    h, w = keep.shape
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [
        cv2.approxPolyDP(c, TRACE_TOLERANCE_PX, True).reshape(-1, 2).tolist()
        for c in sorted(contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(c) >= MIN_AREA * w * h
    ]
    return {"size": [w, h], "polygons": polygons, "keep": round(float(keep.mean()), 4)}


def render(polygons: list, size: tuple[int, int]) -> np.ndarray:
    """The polygons filled white on black at `size` = (w, h); raises ValueError on a bad shape."""
    w, h = size
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("draw at least one polygon")
    shapes = []
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError("a polygon needs at least 3 points")
        try:
            pts = np.array([[float(x), float(y)] for x, y in poly])
        except (TypeError, ValueError):
            raise ValueError("points are [x, y] numbers") from None
        if not np.isfinite(pts).all():
            raise ValueError("points are [x, y] numbers")
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        shapes.append(np.round(pts).astype(np.int32))
    if sum(len(s) for s in shapes) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} points in all")
    img = np.zeros((h, w, 3), np.uint8)
    cv2.fillPoly(img, shapes, (255, 255, 255))
    if not img.any():
        raise ValueError("the polygons cover nothing")
    return img


def _backup(path: Path) -> Path | None:
    if not path.is_file():
        return None
    folder = path.parent / HISTORY_DIR
    folder.mkdir(exist_ok=True)
    stamp = datetime.now()
    dst = folder / f"{path.stem}.{stamp:%Y%m%d-%H%M%S}{path.suffix}"
    while dst.exists():  # two saves within a second: the next second's name
        stamp = stamp.fromtimestamp(stamp.timestamp() + 1)
        dst = folder / f"{path.stem}.{stamp:%Y%m%d-%H%M%S}{path.suffix}"
    shutil.copy2(path, dst)
    return dst


def save(path: Path, polygons: list, size: tuple[int, int], picture: Path | None = None) -> dict:
    """Write the polygons as the mask at `path`, keeping the old file in the history,
    and `picture`, the one it was drawn on, as its reference."""
    img = render(polygons, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup(path)
    tmp = path.with_name(f"{path.stem}.saving{path.suffix}")
    if not cv2.imwrite(str(tmp), img):
        raise OSError(f"cannot write {tmp}")
    tmp.replace(path)
    reference = None
    if picture is not None and picture.is_file():
        reference = path.parent / REFERENCE_DIR / f"{path.stem}{picture.suffix.lower()}"
        reference.parent.mkdir(exist_ok=True)
        shutil.copyfile(picture, reference)
    return {
        "saved": path.name,
        "backup": backup.name if backup else None,
        "reference": reference.name if reference else None,
    }


def history(path: Path) -> list[dict]:
    """Earlier versions of a mask, newest first: {name, saved_at}."""
    folder = path.parent / HISTORY_DIR
    if not folder.is_dir():
        return []
    out = []
    for f in folder.glob(f"{path.stem}.*{path.suffix}"):
        stamp = f.name[len(path.stem) + 1 : -len(path.suffix)]
        try:
            when = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
        except ValueError:
            continue
        out.append({"name": f.name, "saved_at": when.timestamp()})
    return sorted(out, key=lambda e: e["saved_at"], reverse=True)


def restore(path: Path, name: str) -> dict:
    """Put an earlier version back (the current one goes to the history first)."""
    if not any(e["name"] == name for e in history(path)):
        raise ValueError(f"no earlier version {name!r} of {path.name}")
    _backup(path)
    shutil.copy2(path.parent / HISTORY_DIR / name, path)
    return {"saved": path.name, "restored": name}
