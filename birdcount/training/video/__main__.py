"""Count birds over the length of a video: a timeline, its busiest moments, overlays.

    python -m birdcount.training.video ../data/videos/B8A44FD52B7F_20260312_103023_ADA3.mkv
    python -m birdcount.training.video clip.mkv --every 0 --render overlay        # full-rate overlay video
    python -m birdcount.training.video ../data/videos --frames boxes --width 960  # every clip in a folder
    python -m birdcount.training.video day.mkv --every 10 --count-threshold 140

Counted exactly as deployed: frames fitted to IMAGE_WIDTH x IMAGE_HEIGHT, fp16,
the camera's region mask found by the MAC in the filename. The wall clock
comes from the filename when it carries a start stamp (--start-time overrides).

Each video gets a run folder, OUTPUT_ROOT/video/<video>/<time>/ by default;
see `birdcount.training.video.analyze` for what is in it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

from birdcount.core.frames import VIDEO_SUFFIXES
from birdcount.core.inference import DensityPredictor
from birdcount.core.log import setup_logging
from birdcount.core.masks import RegionMasks
from birdcount.core.model import load_inference_model
from birdcount.core.settings import CoreSettings
from birdcount.training.video.analyze import Options, analyze
from birdcount.training.video.render import FRAME_STYLES, VIDEO_MODES
from birdcount.training.video.timeline import parse_start_time


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m birdcount.training.video",
        description="Chart the predicted flock count and the peak local density over a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("input")
    g.add_argument("videos", nargs="+", type=Path, help="video files, or folders of them")
    g.add_argument("--every", type=float, default=1.0, help="seconds between samples; 0 = every frame")
    g.add_argument("--seconds", type=float, default=0.0, help="only the first N seconds (0 = all)")
    g.add_argument(
        "--max-samples", type=int, default=5000, help="at most this many samples per video; --every stretches to fit"
    )
    g.add_argument(
        "--start-time",
        default=None,
        help="wall clock of the first frame ('16:20', '2026-08-28 16:20:00'); default from the filename",
    )

    g = p.add_argument_group("model")
    g.add_argument("--ckpt", type=Path, default=None, help="checkpoint; default MODEL_PATH")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--batch-size", type=int, default=8)
    mask = g.add_mutually_exclusive_group()
    mask.add_argument("--mask", type=Path, default=None, help="region mask to use instead of the camera's own")
    mask.add_argument("--no-mask", action="store_true", help="count the whole frame")

    g = p.add_argument_group("density")
    g.add_argument(
        "--window", type=int, default=16, help="peak window side in density cells (8 px each): 16 = a 128 px patch"
    )
    g.add_argument(
        "--count-threshold", type=float, default=0.0, help="alarm level on the count series (see `deploy --check`)"
    )
    g.add_argument("--peak-threshold", type=float, default=0.0, help="alert level on the peak series")
    g.add_argument(
        "--report",
        choices=("both", "count", "peak"),
        default="both",
        help="series in the figure; 'count' also picks peak frames by count. The files always hold both",
    )
    g.add_argument("--smooth", type=int, default=5, help="samples in the rolling mean drawn over each series")

    g = p.add_argument_group("output")
    g.add_argument(
        "--output-dir", type=Path, default=None, help="run folder (with several videos: one per video under it)"
    )
    g.add_argument(
        "--frames",
        choices=FRAME_STYLES,
        default="none",
        help="also save every sampled frame: 'overlay' with heatmap and count, 'boxes' adding the peak window, "
        "'plain' untouched (for annotating)",
    )
    g.add_argument(
        "--render",
        choices=VIDEO_MODES,
        default="none",
        help="write a video of the sampled frames: heatmap overlay, side by side with the density map, or the "
        "density map alone. It plays in real time; --every 0 gives the full frame rate",
    )
    g.add_argument(
        "--width", type=int, default=0, help="scale saved frames and the video to this width (0 = as analysed)"
    )
    g.add_argument("--peaks", type=int, default=4, help="busiest moments saved as overlays (0 = none)")
    g.add_argument("--peak-gap", type=float, default=15.0, help="minimum seconds between two peak moments")
    g.add_argument("--no-chart", action="store_true", help="skip timeline.png")
    return p


def find_videos(paths: list[Path]) -> list[Path]:
    videos = []
    for path in paths:
        if path.is_dir():
            videos += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            videos.append(path)
        else:
            raise FileNotFoundError(f"no such video or folder: {path}")
    return videos


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.every < 0 or args.window < 1 or args.batch_size < 1:
        parser.error("--every must be >= 0, --window and --batch-size >= 1")
    try:
        videos = find_videos(args.videos)
        start = parse_start_time(args.start_time) if args.start_time else None
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    if not videos:
        parser.error("no videos found")

    settings = CoreSettings.load()
    setup_logging(settings.log_level)
    checkpoint = args.ckpt or settings.model_path
    if checkpoint is None:
        parser.error("no checkpoint: pass --ckpt or set MODEL_PATH")
    frame_size = settings.frame_size
    model = load_inference_model(checkpoint, args.device, input_order="bgr")
    opts = Options(
        every=args.every,
        seconds=args.seconds,
        max_samples=args.max_samples,
        start=start,
        window=args.window,
        thresholds={"count": args.count_threshold, "peak": args.peak_threshold},
        report=args.report,
        smooth=args.smooth,
        frames=args.frames,
        render=args.render,
        width=args.width,
        peaks=args.peaks,
        peak_gap=args.peak_gap,
        chart=not args.no_chart,
    )

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    for video in videos:
        if args.no_mask:
            masks, mask_source = None, None
        elif args.mask is not None:
            masks, mask_source = RegionMasks.from_file(args.mask, frame_size), str(args.mask)
        else:
            masks = RegionMasks.for_source(settings.mask_dir, video, frame_size)
            mask_source = None if masks is None else f"{masks.macs[0]} in {settings.mask_dir}"
        predictor = DensityPredictor(model, frame_size=frame_size, masks=masks, max_batch=args.batch_size)
        if args.output_dir is None:
            out_dir = settings.output_root / "video" / video.stem / stamp
        else:
            out_dir = args.output_dir / video.stem if len(videos) > 1 else args.output_dir
        record = analyze(video, predictor, out_dir, opts, about={"checkpoint": str(checkpoint), "mask": mask_source})
        _print_summary(record, out_dir)
        if record["status"] == "interrupted":
            return 130
    return 0


def _print_summary(record: dict, out_dir: Path) -> None:
    s, cfg = record["summary"], record["settings"]
    count, peak = s["count"], s["peak"]
    print(f"  {s['samples']} samples in {record['seconds']:.1f}s ({record['status']})")
    print(f"  count  max {count['max']:.1f} at {count['max_clock']}, mean {count['mean']:.1f}, p95 {count['p95']:.1f}")
    print(
        f"  peak   max {peak['max']:.2f} birds in one {cfg['window_px']} px patch at {peak['max_clock']}, "
        f"mean {peak['mean']:.2f}, p95 {peak['p95']:.2f}"
    )
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
