"""Make distillation pairs for compression-robust training (`train --distill`).

    # training pairs: windows of a replay's camera-days, the review clips' days left out
    python -m birdcount.training.distill outputs/distill/train \\
        --samples outputs/replay/full/samples --alarm-config configs/alarm.json \\
        --exclude-days-of ../camera_videos/selected_dense_video_review_20260716
    # held-out pairs: the review clips themselves, at fixed deployment-like settings
    python -m birdcount.training.distill outputs/distill/holdout --holdout \\
        --clips "../camera_videos/selected_dense_video_review_20260716/**/*_original.mkv" \\
        --alarm-config configs/alarm.json

The teacher is MODEL_PATH (or --teacher). Thresholds come from the alarm
config only to tell crowded windows and frames; nothing else of it is used.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

from birdcount.core.camera_identity import extract_mac


def _thresholds(path: Path) -> dict[str, float]:
    cams = json.loads(Path(path).read_text(encoding="utf-8")).get("cameras", [])
    out = {}
    for c in cams:
        mac = extract_mac(str(c["camera_id"]))
        if mac:
            out[mac] = float(c["threshold"])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m birdcount.training.distill", description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path, help="pairs folder (appended to if it exists)")
    p.add_argument("--alarm-config", type=Path, required=True, help="per-camera thresholds, to tell crowded frames")
    p.add_argument("--samples", type=Path, help="a replay's samples/ folder: the windows to draw from")
    p.add_argument("--exclude-days-of", type=Path, action="append", default=[],
                   help="leave out the camera-days of the clips under this folder (the test set)")  # fmt: skip
    p.add_argument("--holdout", action="store_true", help="fixed settings on --clips instead of random ones")
    p.add_argument("--clips", help="--holdout: a glob of the clips")
    p.add_argument("--teacher", type=Path, help="default MODEL_PATH")
    p.add_argument("--dense-per-day", type=int, default=3)
    p.add_argument("--random-per-day", type=int, default=1)
    p.add_argument("--per-window", type=int, default=10, help="pairs kept per window")
    p.add_argument("--window-seconds", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, help="only the first N windows (a trial)")
    args = p.parse_args(argv)
    if args.holdout == bool(args.samples) or (args.holdout and not args.clips):
        p.error("give --samples (training pairs) or --holdout with --clips")

    from birdcount.core.log import setup_logging
    from birdcount.core.model import load_inference_model
    from birdcount.core.settings import CoreSettings
    from birdcount.training.distill.pairs import (
        HOLDOUT_SETTINGS,
        Teacher,
        Window,
        camera_days,
        random_setting,
        windows_from_replay,
        write_pairs,
    )

    s = CoreSettings.load()
    setup_logging(s.log_level)
    thresholds = _thresholds(args.alarm_config)
    teacher_path = args.teacher or s.model_path
    if teacher_path is None:
        p.error("no teacher: set MODEL_PATH or give --teacher")
    model = load_inference_model(teacher_path, "cuda", input_order="bgr")
    teacher = Teacher(model, s.frame_size, s.mask_dir)
    rng = random.Random(args.seed)

    if args.holdout:
        clips = sorted(Path(c) for c in glob.glob(args.clips, recursive=True))
        jobs = [
            (Window(mac, clip, offset), setting)
            for clip in clips
            if (mac := extract_mac(str(clip)))
            for offset in (30.0, 150.0)
            for setting in HOLDOUT_SETTINGS
        ]
        prefix = "h"
    else:
        exclude = camera_days(q for d in args.exclude_days_of for q in Path(d).rglob("*") if q.is_file())
        windows = windows_from_replay(
            args.samples,
            thresholds,
            exclude=exclude,
            rng=rng,
            dense_per_day=args.dense_per_day,
            random_per_day=args.random_per_day,
            window_s=args.window_seconds,
        )
        jobs = [(w, random_setting(rng)) for w in windows]
        prefix = "w"
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} windows -> {args.out}", flush=True)
    n = write_pairs(
        args.out, jobs, teacher, thresholds=thresholds, per_window=args.per_window,
        window_s=args.window_seconds, workers=args.workers, prefix=prefix,
    )  # fmt: skip
    print(f"{n} pairs written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
