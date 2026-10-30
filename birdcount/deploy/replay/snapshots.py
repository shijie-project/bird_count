"""Pictures of a replayed camera at every crowding level: what is kept of the recordings.

    python -m birdcount.deploy.replay.snapshots B8A44FD51C3C          # into DATA_ROOT/snapshots/B8A44FD51C3C/

The recordings stay where they are; what later work needs from them is a
picture of each crowding level the camera goes through, with the count it was
given. Picked from the run's samples (no counting again) and decoded from the
recordings, per day:

  bin    per 10-bird bin from `--floor` up, `--per-bin` moments at least
         `--gap` minutes apart, drawn at random over the day (seeded by camera
         and day, so the same run picks the same moments again)
  peak   the day's busiest moments: each high point of the 2 s median count
         (the alarm's) from `--peak-floor` up, at least `--peak-gap` minutes apart

Written twice:

  DATA_ROOT/snapshots/<MAC>/<day>/<HHMMSS>_<why>_<count>.jpg
         the frame as the model saw it (1280x720, unmasked), and index.csv:
         each picture's time, count, 2 s median, clip and frame, and the model
         and mask that counted it. For counting again under another mask or
         model, rating, labelling.
  the run's count log (count_log/<day>/snapshots/<MAC>/, 640x360, named
         the same), replacing the count logger's: the farm backend shows them
         and deploy.calibrate rates them (COUNT_LOG_DIR=<run>/count_log, --days N).

Exit codes: 0 done, 2 nothing to do or a setup error.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import shutil
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


logger = logging.getLogger("birdcount.deploy.replay.snapshots")

JPEG_QUALITY = 95  # the kept frame: counted again later, so close to the decoded one
SMOOTHING_SECONDS = 2.0  # the alarm's median window (configs/alarm.json `smoothing_seconds`)
INDEX_FIELDS = [
    "file",
    "mac",
    "day",
    "time_local",
    "timestamp",
    "count",
    "count_2s",
    "why",
    "clip",
    "frame",
    "model_sha256",
    "mask",
]


@dataclass(frozen=True)
class Rules:
    floor: int = 40  # lowest count bin pictured
    per_bin: int = 3  # pictures per bin and day
    gap: float = 20 * 60.0  # s between two pictures of one bin
    peak_floor: float = 80.0  # lowest 2 s median that is a peak
    peak_gap: float = 10 * 60.0  # s between two peaks


@dataclass(frozen=True)
class Pick:
    i: int  # sample index in the day
    why: str  # bin / peak


def smoothed(count, fps: float):
    """The 2 s median the alarm compares, over consecutive samples (a hole in a recording is not bridged)."""
    import numpy as np

    n = max(1, round(SMOOTHING_SECONDS * fps))
    if len(count) < n:
        return count.astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(count, n)
    return np.concatenate([count[: n - 1], np.median(windows, axis=1)]).astype(np.float32)


def pick(t, count, smooth, rules: Rules, seed: int) -> list[Pick]:
    """The day's moments to picture (see the module doc), in time order."""
    import numpy as np

    peaks: list[int] = []
    for i in np.argsort(-smooth, kind="stable"):
        if smooth[i] < rules.peak_floor:
            break
        if all(abs(t[i] - t[j]) >= rules.peak_gap for j in peaks):
            peaks.append(int(i))
    taken = set(peaks)
    picks = [Pick(i, "peak") for i in peaks]
    rng = np.random.default_rng(seed)
    bins = (np.maximum(count, 0) // 10).astype(np.int64)
    for b in range(rules.floor // 10, int(bins.max(initial=0)) + 1):
        chosen: list[int] = []
        for i in rng.permutation(np.nonzero(bins == b)[0]):
            if len(chosen) == rules.per_bin:
                break
            if i not in taken and all(abs(t[i] - t[j]) >= rules.gap for j in chosen):
                chosen.append(int(i))
        picks += [Pick(i, "bin") for i in chosen]
    return sorted(picks, key=lambda p: p.i)


def name_of(t: float, why: str, count: float) -> str:
    """A picture's file name: the count log's snapshot naming (countlog.SNAPSHOT_RE)."""
    return f"{datetime.fromtimestamp(t):%H%M%S}_{why}_{count:.0f}.jpg"


# --- one camera ---------------------------------------------------------------------------------------


def snapshot_camera(layout, mac: str, out_root: Path, rules: Rules, *, jobs: int, copy_ahead: int) -> int:
    """Pick, decode and write every day's pictures of `mac`; returns how many."""
    import numpy as np

    from birdcount.deploy.replay.cache import ClipCache, InPlace

    about = json.loads((layout.root / "run.json").read_text(encoding="utf-8"))
    size = tuple(about["frame_size"])
    days = [p.stem for p in sorted((layout.samples / mac).glob("*.json"))]
    camera_dir = out_root / mac
    shutil.rmtree(camera_dir, ignore_errors=True)  # picked again from the same run: the same moments
    cache = ClipCache(layout.root / "cache", ahead=copy_ahead) if copy_ahead else None
    rows: list[dict] = []
    lock = threading.Lock()

    def one_day(day: str) -> None:
        record = layout.record(mac, day)
        with np.load(layout.samples_of(mac, day)) as z:
            s = {k: z[k] for k in ("t", "count", "clip", "frame", "clips")}
        smooth = smoothed(s["count"], record["fps"])
        picks = pick(s["t"], s["count"], smooth, rules, zlib.crc32(f"{mac} {day}".encode()))
        by_clip: dict[int, list[Pick]] = {}
        for p in picks:
            by_clip.setdefault(int(s["clip"][p.i]), []).append(p)
        clips = sorted(by_clip)
        sources = [Path(str(s["clips"][c])) for c in clips]
        plan = cache.open(sources) if cache else InPlace(sources)
        frames = {}
        try:
            for k, c in enumerate(clips):
                try:
                    got = _frames_at(plan.fetch(k), [int(s["frame"][p.i]) for p in by_clip[c]], size)
                except OSError as e:
                    logger.warning("[%s %s] no pictures from %s: %s", mac, day, sources[k].name, e)
                    got = {}
                finally:
                    plan.done(k)
                for p in by_clip[c]:
                    frames[p.i] = got.get(int(s["frame"][p.i]))
        finally:
            plan.close()
        written = _write_day(layout, mac, day, record, s, smooth, picks, frames, camera_dir, about, rules.floor)
        with lock:
            rows.extend(written)
        logger.info("[%s %s] %d picture(s)", mac, day, len(written))

    try:
        with ThreadPoolExecutor(max(1, jobs)) as pool:
            list(pool.map(one_day, days))
    finally:
        if cache is not None:
            cache.close()
    rows.sort(key=lambda r: r["timestamp"])
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    camera_dir.mkdir(parents=True, exist_ok=True)
    (camera_dir / "index.csv").write_text(buf.getvalue(), encoding="utf-8")
    return len(rows)


def _frames_at(path: Path, indices: list[int], size: tuple[int, int]) -> dict:
    from birdcount.core.video import frames_at

    return frames_at(path, indices, size)


def _write_day(layout, mac, day, record, s, smooth, picks, frames, camera_dir, about, floor) -> list[dict]:
    """The day's pictures into the data folder and the run's count log; the count log's rows point at them."""
    import cv2

    from birdcount.deploy.handlers.count_logger import JPEG_QUALITY as SMALL_QUALITY
    from birdcount.deploy.handlers.count_logger import SNAPSHOT_SIZE

    day_dir = camera_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    small_dir = layout.count_log / day / "snapshots" / mac
    shutil.rmtree(small_dir, ignore_errors=True)  # the count logger's, picked by the threshold of the day
    small_dir.mkdir(parents=True)
    mask = record.get("meta", {}).get("masks", {}).get(mac, "")
    rows, by_second = [], {}
    for p in picks:
        frame = frames.get(p.i)
        if frame is None:
            continue
        t, count = float(s["t"][p.i]), float(s["count"][p.i])
        name = name_of(t, p.why, count)
        cv2.imwrite(str(day_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        small = cv2.resize(frame, SNAPSHOT_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(small_dir / name), small, [int(cv2.IMWRITE_JPEG_QUALITY), SMALL_QUALITY])
        by_second[int(t)] = f"snapshots/{mac}/{name}"
        rows.append(
            {
                "file": f"{mac}/{day}/{name}",
                "mac": mac,
                "day": day,
                "time_local": f"{datetime.fromtimestamp(t):%Y-%m-%d %H:%M:%S.%f}"[:-3],
                "timestamp": f"{t:.3f}",
                "count": f"{count:.2f}",
                "count_2s": f"{float(smooth[p.i]):.2f}",
                "why": p.why,
                "clip": str(s["clips"][s["clip"][p.i]]),
                "frame": int(s["frame"][p.i]),
                "model_sha256": about.get("model_sha256", ""),
                "mask": mask,
            }
        )
    _repoint(layout.count_log / day / f"{mac}.csv", by_second)
    meta = record.setdefault("meta", {})
    meta["snapshot_floors"] = {**meta.get("snapshot_floors", {}), mac: floor}
    layout.finish(mac, day, record)
    return rows


def _repoint(path: Path, by_second: dict[int, str]) -> None:
    """The count log's `snapshot` column: the picture taken in each second, none elsewhere."""
    from birdcount.deploy.countlog import ROW_FIELDS

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["snapshot"] = by_second.get(int(row["timestamp"]), "")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


# --- the command --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    d = Rules()
    p = argparse.ArgumentParser(
        prog="python -m birdcount.deploy.replay.snapshots",
        description="Pictures of a replayed camera at every crowding level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("camera", help="the camera's MAC")
    p.add_argument("--run", default="full", help="run folder under OUTPUT_ROOT/replay/")
    p.add_argument("--out", type=Path, default=None, help="default DATA_ROOT/snapshots")
    p.add_argument("--floor", type=int, default=d.floor, help="lowest count bin pictured")
    p.add_argument("--per-bin", type=int, default=d.per_bin, help="pictures per 10-bird bin and day")
    p.add_argument("--gap", type=float, default=d.gap / 60, help="minutes between two pictures of one bin")
    p.add_argument("--peak-floor", type=float, default=d.peak_floor, help="lowest 2 s median that is a peak")
    p.add_argument("--peak-gap", type=float, default=d.peak_gap / 60, help="minutes between two peaks")
    p.add_argument("--jobs", type=int, default=4, help="days at once")
    p.add_argument("--copy-ahead", type=int, default=2, help="clips copied off a slow disk ahead; 0 = none")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mac = args.camera.strip().upper()

    from birdcount.core.log import setup_logging
    from birdcount.deploy.replay.layout import RunLayout
    from birdcount.deploy.settings import DeploySettings

    settings = DeploySettings.load()
    setup_logging(settings.log_level)
    layout = RunLayout(settings.output_root / "replay" / args.run)
    if not (layout.root / "run.json").is_file():
        parser.error(f"no replay in {layout.root}")
    if not any((layout.samples / mac).glob("*.json")):
        logger.error("No camera-days of %s counted in %s.", mac, layout.root)
        return 2
    rules = Rules(args.floor, args.per_bin, args.gap * 60, args.peak_floor, args.peak_gap * 60)
    out = args.out or settings.data_root / "snapshots"
    n = snapshot_camera(layout, mac, out, rules, jobs=args.jobs, copy_ahead=args.copy_ahead)
    logger.info("%d picture(s) of %s in %s (index.csv), and in the run's count log.", n, mac, out / mac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
