"""Replay recorded video through the deployed pipeline, camera-day by camera-day.

    python -m birdcount.deploy.replay G:/ --list                              # the cameras and days found
    python -m birdcount.deploy.replay G:/ --camera B8A44FD51C3C               # every day of one camera
    python -m birdcount.deploy.replay G:/ --camera B8A44FD51C3C --days 2026-03-11,2026-03-14..2026-03-16
    python -m birdcount.deploy.replay G:/ --camera all --jobs 6
    python -m birdcount.deploy.webui --replay outputs/replay/full             # look at the result

Counted as deployed (MODEL_PATH, MASK_DIR, IMAGE_WIDTH x IMAGE_HEIGHT, FPS and
ALARM_CONFIG_PATH from .env) into OUTPUT_ROOT/replay/<run>/ (see `layout`).
Camera-days already done are skipped, so a replay that was stopped continues
where it stopped; --force counts them again.

`--jobs` camera-days run at once, each decoding its own clips on the CPU,
which is what limits a single one (~320 samples/s). Read by several at once a
USB hard disk seeks itself down to a third of its speed, so the clips are
copied to a local folder one after the other, `--copy-ahead` clips ahead of
each camera-day (see `cache`); --copy-ahead 0 reads them in place, for
recordings already on a fast disk.

Exit codes: 0 done, 1 a camera-day failed, 2 nothing to do or a setup error, 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from datetime import date, timedelta
from pathlib import Path

from birdcount.deploy.replay.recordings import Camera, days_of, find_cameras


logger = logging.getLogger("birdcount.deploy.replay")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m birdcount.deploy.replay",
        description="Replay recorded video through the deployed pipeline, camera-day by camera-day.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "root", type=Path, help="the recordings: the archive, one recorder's axisN or one axis-<MAC> folder"
    )
    p.add_argument("--list", action="store_true", help="list the cameras and days found, and exit")
    p.add_argument("--camera", default=None, help="MACs, comma-separated, or 'all'")
    p.add_argument(
        "--days", default=None, help="YYYY-MM-DD days and FROM..TO ranges, comma-separated; default every day"
    )
    p.add_argument("--run", default="full", help="run folder under OUTPUT_ROOT/replay/")
    p.add_argument("--jobs", type=int, default=4, help="camera-days counted at once")
    p.add_argument(
        "--copy-ahead", type=int, default=3, help="clips copied to a local folder ahead of each camera-day; 0 = none"
    )
    p.add_argument("--batch-size", type=int, default=8, help="frames per model batch")
    p.add_argument("--force", action="store_true", help="count camera-days again that the run has done")
    p.add_argument("--ckpt", type=Path, default=None, help="checkpoint; default MODEL_PATH")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alarm-config", type=Path, default=None, help="alarm config; default ALARM_CONFIG_PATH")
    return p


def parse_days(spec: str) -> set[str]:
    """'2026-03-11,2026-03-14..2026-03-16' -> those four days."""
    out = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        first, _, last = part.partition("..")
        a = date.fromisoformat(first)
        b = date.fromisoformat(last) if last else a
        if b < a:
            raise ValueError(f"range {part!r} runs backwards")
        out |= {f"{a + timedelta(days=i)}" for i in range((b - a).days + 1)}
    return out


def _list(cameras: dict[str, Camera]) -> None:
    print(f"{'MAC':<13} {'recorder':<9} {'days':>4} {'clips':>6}  first .. last")
    for mac, cam in cameras.items():
        days = days_of(cam)
        clips = sum(len(c) for c in days.values())
        span = f"{min(days)} .. {max(days)}" if days else "-"
        print(f"{mac:<13} {cam.group:<9} {len(days):>4} {clips:>6}  {span}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.batch_size < 1 or args.copy_ahead < 0:
        parser.error("--jobs and --batch-size must be at least 1, --copy-ahead at least 0")
    if not args.root.is_dir():
        parser.error(f"no such folder: {args.root}")
    cameras = find_cameras(args.root)
    if not cameras:
        parser.error(f"no axis-<MAC> camera folders under {args.root}")
    if args.list:
        _list(cameras)
        return 0
    if not args.camera:
        parser.error("--camera is required (or --list)")
    macs = (
        list(cameras) if args.camera.strip().lower() == "all" else [m.strip().upper() for m in args.camera.split(",")]
    )
    unknown = [m for m in macs if m not in cameras]
    if unknown:
        parser.error(f"no recordings of {', '.join(unknown)} under {args.root} (see --list)")
    try:
        wanted = parse_days(args.days) if args.days else None
    except ValueError as e:
        parser.error(f"--days: {e}")

    from birdcount.core.log import setup_logging
    from birdcount.core.masks import mask_digests
    from birdcount.deploy.alarm import load_config
    from birdcount.deploy.countlog import file_sha256
    from birdcount.deploy.replay.day import DayJob
    from birdcount.deploy.replay.layout import RunLayout, RunMismatch
    from birdcount.deploy.settings import DeploySettings

    settings = DeploySettings.load()
    layout = RunLayout(settings.output_root / "replay" / args.run)
    layout.root.mkdir(parents=True, exist_ok=True)
    setup_logging(settings.log_level, log_file=layout.root / "replay.log")
    checkpoint = args.ckpt or settings.model_path
    if checkpoint is None or not Path(checkpoint).is_file():
        parser.error(f"no checkpoint: {checkpoint or 'pass --ckpt or set MODEL_PATH'}")
    config = load_config(args.alarm_config or settings.alarm_config_path)

    jobs, skipped = [], 0
    for mac in macs:
        for day, clips in days_of(cameras[mac]).items():
            if wanted is not None and day not in wanted:
                continue
            if layout.done(mac, day) and not args.force:
                skipped += 1
                continue
            jobs.append(DayJob(mac, day, tuple(clips)))
    if skipped:
        logger.info("%d camera-day(s) already done in %s; --force counts them again.", skipped, layout.root)
    if not jobs:
        logger.info("Nothing to count.")
        return 2

    about = {
        "model_path": str(checkpoint),
        "model_sha256": file_sha256(Path(checkpoint)),
        "frame_size": list(settings.frame_size),
        "fps": settings.fps,
        "mask_dir": str(settings.mask_dir or ""),
        "alarm_config": str(args.alarm_config or settings.alarm_config_path),
    }
    try:
        layout.open(about)
    except RunMismatch as e:
        logger.error("%s", e)
        return 2
    meta = {
        "frame_size": list(settings.frame_size),
        "fps": settings.fps,
        "mask_dir": str(settings.mask_dir or ""),
        "masks": mask_digests(settings.mask_dir),
        "source": "replay",
        "recordings": str(args.root),
    }
    logger.info(
        "%d camera-day(s) of %d camera(s), %d clips, %d at a time -> %s",
        len(jobs),
        len({j.mac for j in jobs}),
        sum(len(j.clips) for j in jobs),
        args.jobs,
        layout.root,
    )
    return _run(jobs, layout, config, settings, meta, checkpoint, args)


def _run(jobs, layout, config, settings, meta, checkpoint, args) -> int:
    from birdcount.core.inference import DensityPredictor
    from birdcount.core.masks import RegionMasks
    from birdcount.core.model import load_inference_model
    from birdcount.deploy.replay.cache import ClipCache
    from birdcount.deploy.replay.day import Outputs, replay_day

    model = load_inference_model(checkpoint, args.device, input_order="bgr")
    cache = ClipCache(layout.root / "cache", ahead=args.copy_ahead) if args.copy_ahead else None
    pending: queue.Queue = queue.Queue()
    for job in jobs:
        pending.put(job)
    stop = threading.Event()
    failed: list[str] = []
    done: list[str] = []

    def predictor_for(mac: str) -> DensityPredictor:
        masks = RegionMasks.from_dir(settings.mask_dir, [mac], settings.frame_size, name=mac)
        return DensityPredictor(model, frame_size=settings.frame_size, masks=masks, max_batch=args.batch_size)

    def work() -> None:
        mac, predictor = None, None
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            name = f"{job.mac} {job.day}"
            try:
                if job.mac != mac:
                    mac, predictor = job.mac, predictor_for(job.mac)
                layout.clear(job.mac, job.day)
                logger.info(
                    "[%s] %d clips, %s .. %s",
                    name,
                    len(job.clips),
                    *(f"{c.start:%H:%M}" for c in (job.clips[0], job.clips[-1])),
                )
                out = Outputs(
                    layout.count_log, layout.evidence_of(job.mac, job.day), layout.samples_of(job.mac, job.day)
                )
                source = cache.open([c.path for c in job.clips]) if cache else None
                try:
                    record = replay_day(
                        job,
                        predictor,
                        config,
                        out,
                        fps=settings.fps,
                        meta=meta,
                        model_path=Path(checkpoint),
                        stop=stop,
                        source=source,
                    )
                finally:
                    if source is not None:
                        source.close()
                if record is None:
                    return
                layout.finish(job.mac, job.day, record)
                done.append(name)
                logger.info("[%s] done: %s", name, _describe(record))
            except Exception:
                logger.exception("[%s] failed.", name)
                failed.append(name)

    threads = [
        threading.Thread(target=work, name=f"Replay-{i}", daemon=True) for i in range(min(args.jobs, len(jobs)))
    ]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:
        logger.warning("Stopping: the camera-days in progress are dropped and counted again next time.")
        stop.set()
        for t in threads:
            t.join()
        return 130
    finally:
        if cache is not None:
            cache.close()
    logger.info(
        "%d camera-day(s) done, %d failed%s.", len(done), len(failed), f": {', '.join(failed)}" if failed else ""
    )
    logger.info("Look at it: python -m birdcount.deploy.webui --replay %s", layout.root)
    return 1 if failed else 0


def _describe(record: dict) -> str:
    c, over, alarms = record["count"], record["seconds_over"], record["alarms"]
    failed = sum(1 for clip in record["clips"] if clip.get("error"))
    text = (
        f"{record['hours']:.1f} h in {record['counted_in_s']:.0f} s ({record['samples_per_s']:.0f} samples/s); "
        f"count mean {c['mean']} p95 {c['p95']} max {c['max']}"
    )
    if record["threshold"] is not None:
        text += f"; T {record['threshold']:g}: {over.get('1.0', 0)} s at or over, Level 1 x{alarms.get('level1', 0)}"
    if failed:
        text += f"; {failed} clip(s) unreadable"
    return text


if __name__ == "__main__":
    sys.exit(main())
