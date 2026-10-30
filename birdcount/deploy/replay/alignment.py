"""Does a camera's region mask fit the pictures it is applied to?

    python -m birdcount.deploy.replay.alignment B8A44FD51C3C
    python -m birdcount.deploy.replay.alignment B8A44FD51C3C --against ../data/stills/2026-10-31_1000/B8A44FD51C3C.jpg
    python -m birdcount.deploy.replay.alignment B8A44F5B4394 --recordings G:/    # a camera not replayed yet

A mask is pixels: a camera bumped, re-aimed or zoomed since it was drawn
counts the wrong part of the shed, and a threshold tuned on those counts is
off. Each replayed day's emptiest pictures (DATA_ROOT/snapshots/<MAC>/, where
the floor and the fittings show most), or with `--recordings` a frame of each
recorded day around noon straight from the clips (no replay needed), are
registered against a reference: `--against` (e.g. a still of today,
deploy.tools.stills), else the picture the mask was drawn on
(MASK_DIR/_reference/, kept by the farm backend's Masks page), else a frame of
the camera's clip in DATA_ROOT/videos/ (what masks were drawn on before the
page kept pictures), else the first day. Every still of the camera in
DATA_ROOT/stills/ is registered too, unless dark (lights off).

Registration matches features of the fixed fittings (ORB, ratio test) and
fits a shift + rotation + scale with RANSAC. Features are looked for outside
the mask, where the walls, fences and feeders are and the birds mostly are
not (the whole frame when that leaves too little); birds that move between
pictures fall out as outliers. A picture off by more than `--max-shift` px, 0.5° or
1 % is flagged MOVED: the mask does not fit it as it fits the reference.
Writes alignment.csv and alignment.jpg (each picture with the mask's edge
drawn on) next to the camera's snapshots.

Exit codes: 0 every picture fits, 1 some moved or could not be registered, 2 setup error.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


MAX_ANGLE = 0.5  # degrees
MAX_SCALE = 0.01
MIN_OUTSIDE = 0.1  # share of the frame outside the mask worth looking at alone
MIN_INLIERS = 25  # fewer matched fittings than this: the fit says nothing
DARK = 25.0  # mean grey level of a still taken with the lights off
TILE = (320, 180)
NOON = 12 * 60  # minutes: the frames taken from recordings are the ones nearest midday, the shed fully lit
CLIP_FRAME = 250  # 10 s into a clip: past any start-of-file glitch
TRIES = 3  # pictures of a day tried; the best fit counts (dawn light or a crowd can hide the fittings in one)


@dataclass(frozen=True)
class Fit:
    dx: float
    dy: float
    angle: float  # degrees
    scale: float
    inliers: int

    @property
    def shift(self) -> float:
        return math.hypot(self.dx, self.dy)

    def verdict(self, max_shift: float) -> str:
        if self.inliers < MIN_INLIERS:
            return "UNSURE"
        moved = self.shift > max_shift or abs(self.angle) > MAX_ANGLE or abs(self.scale - 1) > MAX_SCALE
        return "MOVED" if moved else "ok"


def register(reference, picture, where=None) -> Fit | None:
    """How `picture` (BGR) sits against `reference`: None when no fit can be made.

    `where` (uint8, non-zero = look here) limits the features to part of the
    frame: outside the mask, the walls and fittings the birds do not cover.
    """
    import cv2
    import numpy as np

    def features(img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
        return cv2.ORB_create(nfeatures=4000).detectAndCompute(grey, where)

    ka, da = features(reference)
    kb, db = features(picture)
    if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return None
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return Fit(float(matrix[0, 2]), float(matrix[1, 2]), angle, scale, int(inliers.sum()))


def reference_picture(mask_dir: Path | None, mac: str) -> Path | None:
    """The picture the camera's current mask was drawn on, when the Masks page kept it."""
    from birdcount.core.masks import REFERENCE_DIR

    if mask_dir is None or not (mask_dir / REFERENCE_DIR).is_dir():
        return None
    found = sorted((mask_dir / REFERENCE_DIR).glob(f"*{mac}*.jpg"))
    return found[-1] if found else None


def recorded_days(root: Path, mac: str, cache: Path, size: tuple[int, int]) -> dict[str, list[Path]]:
    """A frame of each recorded day, from the `TRIES` clips nearest noon (kept in `cache`)."""
    from birdcount.core.video import frames_at
    from birdcount.deploy.replay.recordings import days_of, find_cameras

    camera = find_cameras(root).get(mac)
    if camera is None:
        return {}
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[Path]] = {}
    for day, clips in days_of(camera).items():
        for clip in sorted(clips, key=lambda c: abs(c.start.hour * 60 + c.start.minute - NOON))[:TRIES]:
            path = cache / f"{day}_{clip.start:%H%M}.jpg"
            if not path.is_file():
                try:  # a clip shorter than CLIP_FRAME gives its first frame
                    got = frames_at(clip.path, [CLIP_FRAME], size).get(CLIP_FRAME)
                    got = got if got is not None else frames_at(clip.path, [0], size).get(0)
                except OSError:  # an empty or broken file
                    got = None
                if got is None:
                    continue
                _write(path, got)
            out.setdefault(day, []).append(path)
    return out


def clip_reference(data_root: Path, mac: str, cache: Path, size: tuple[int, int]) -> Path | None:
    """A frame of the camera's clip in DATA_ROOT/videos: the picture its mask was drawn on, as masks were made."""
    from birdcount.core.paths import VIDEO_SUFFIXES
    from birdcount.core.video import frames_at

    clips = sorted(p for p in (data_root / "videos").glob(f"{mac}_*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        return None
    path = cache / f"videos_{clips[-1].stem}.jpg"
    if not path.is_file():
        got = frames_at(clips[-1], [CLIP_FRAME], size).get(CLIP_FRAME)
        if got is None:
            return None
        cache.mkdir(parents=True, exist_ok=True)
        _write(path, got)
    return path


def _write(path: Path, frame) -> None:
    import cv2

    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def emptiest_per_day(snapshots: Path, n: int = TRIES) -> dict[str, list[Path]]:
    """Each day's `n` lowest-count bin pictures: the most floor and fittings showing."""
    out: dict[str, list[tuple[float, Path]]] = {}
    with (snapshots / "index.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["why"] == "bin":
                out.setdefault(row["day"], []).append((float(row["count"]), snapshots.parent / row["file"]))
    return {day: [path for _, path in sorted(found)[:n]] for day, found in sorted(out.items())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m birdcount.deploy.replay.alignment", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("camera", help="the camera's MAC")
    p.add_argument("--against", type=Path, default=None, help="reference picture (default: the mask's, else day one)")
    p.add_argument(
        "--recordings", type=Path, default=None, help="take each day's frame from these recordings (e.g. G:/)"
    )
    p.add_argument("--max-shift", type=float, default=6.0, help="pixels (at 1280x720) a picture may be off")
    args = p.parse_args(argv)
    mac = args.camera.strip().upper()

    import cv2
    import numpy as np

    from birdcount.core.frames import read_image
    from birdcount.core.masks import RegionMasks
    from birdcount.deploy.settings import DeploySettings

    settings = DeploySettings.load()
    size = settings.frame_size
    snapshots = settings.data_root / "snapshots" / mac
    cache = snapshots / "alignment"
    if args.recordings is not None:
        days = recorded_days(args.recordings, mac, cache, size)
    elif (snapshots / "index.csv").is_file():
        days = emptiest_per_day(snapshots)
    else:
        print(f"error: no pictures of {mac} in {snapshots}; pass --recordings G:/ or replay it", file=sys.stderr)
        return 2
    stills = sorted((settings.data_root / "stills").glob(f"*/{mac}.jpg"))
    reference = (
        args.against
        or reference_picture(settings.mask_dir, mac)
        or clip_reference(settings.data_root, mac, cache, size)
        or (next(iter(days.values()))[0] if days else None)
    )
    if reference is None:
        print(f"error: nothing to compare {mac} against", file=sys.stderr)
        return 2
    ref = cv2.resize(read_image(reference), size)
    masks = RegionMasks.from_dir(settings.mask_dir, [mac], size)
    edge = None
    if masks is not None:
        edge = cv2.morphologyEx(masks.keep[0] * 255, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    print(f"{mac}: reference {reference}" + ("" if masks is not None else "  (no mask in MASK_DIR)"))

    snapshots.mkdir(parents=True, exist_ok=True)
    items = list(days.items())
    for still in stills:
        img = read_image(still)
        if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) >= DARK:
            items.append((f"still {still.parent.name}", [still]))
        else:
            print(f"  still {still.parent.name}: dark, skipped")

    outside = None
    if masks is not None and (masks.keep[0] == 0).mean() > MIN_OUTSIDE:
        outside = cv2.dilate(1 - masks.keep[0], np.ones((15, 15), np.uint8)) * 255  # a margin over the edge

    def fit_of(img):
        fit = register(ref, img, outside) if outside is not None else None
        return fit if fit is not None and fit.inliers >= MIN_INLIERS else register(ref, img)

    if not items:
        print(f"error: no pictures of {mac} to compare", file=sys.stderr)
        return 2
    rows, tiles, worst = [], [], 0
    for label, paths in items:
        tried = [(fit_of(img), path, img) for path in paths for img in [cv2.resize(read_image(path), size)]]
        fit, path, img = max(tried, key=lambda x: x[0].inliers if x[0] else -1)
        verdict = fit.verdict(args.max_shift) if fit else "UNSURE"
        worst = max(worst, verdict != "ok")
        text = (
            f"{fit.dx:+6.1f} {fit.dy:+6.1f} px  {fit.angle:+5.2f} deg  x{fit.scale:.3f}  {fit.inliers:4d} matched"
            if fit
            else "no fit"
        )
        print(f"  {label:<22} {text}   {verdict}")
        rows.append(
            {"picture": label, "file": str(path), "verdict": verdict}
            | (
                {
                    "dx": f"{fit.dx:.1f}",
                    "dy": f"{fit.dy:.1f}",
                    "angle": f"{fit.angle:.2f}",
                    "scale": f"{fit.scale:.4f}",
                    "inliers": fit.inliers,
                }
                if fit
                else {}
            )
        )
        tile = img.copy()
        if edge is not None:
            tile[edge] = (0, 0, 255)
        colour = (0, 220, 0) if verdict == "ok" else (0, 0, 255)
        cv2.putText(tile, f"{label}  {verdict}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 4)
        tiles.append(cv2.resize(tile, TILE, interpolation=cv2.INTER_AREA))

    with (snapshots / "alignment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["picture", "file", "verdict", "dx", "dy", "angle", "scale", "inliers"])
        writer.writeheader()
        writer.writerows(rows)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 5]) for i in range(0, len(tiles), 5)])
    cv2.imwrite(str(snapshots / "alignment.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    moved = sum(r["verdict"] == "MOVED" for r in rows)
    unsure = sum(r["verdict"] == "UNSURE" for r in rows)
    print(f"\n{len(rows) - moved - unsure} fit, {moved} MOVED, {unsure} unsure -> {snapshots / 'alignment.jpg'}")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
