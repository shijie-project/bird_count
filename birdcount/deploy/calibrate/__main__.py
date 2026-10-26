"""Suggest each camera's threshold from how crowded its pictures look.

    python -m birdcount.deploy.calibrate rate --dry-run   # what would be sent, and roughly what it costs
    python -m birdcount.deploy.calibrate rate             # rate new snapshots (Batch API; results within hours)
    python -m birdcount.deploy.calibrate rate --now       # one request per picture: seconds, full price
    python -m birdcount.deploy.calibrate propose          # ratings -> a suggested threshold per camera
    python -m birdcount.deploy.calibrate run              # rate, then propose (the nightly job)

`rate` sends the count log's snapshots and the alarm pictures of the last
--days days to a vision model (CALIBRATION_MODEL, key ANTHROPIC_API_KEY),
the area outside each camera's region mask shaded, and stores one crowding
level 1-5 per picture in that day's ratings.csv. Each camera's 10-bird count
bin is rated until it holds 10 ratings, so a camera's common counts stop
costing anything after a day or two.

`propose` turns the ratings into a threshold per camera (see fit.py) and
writes CALIBRATION_DIR/latest.json (plus a dated copy) for the farm
backend's Thresholds page. Nothing changes configs/alarm.json.

Schedule `run` nightly, e.g. with Windows Task Scheduler:
    schtasks /create /tn birdcount-calibrate /sc daily /st 21:00 /tr "<python> -m birdcount.deploy.calibrate run"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


SNAPSHOT_TOKENS = 320  # a 640x360 picture, by the API's width x height / 750 rule
PROMPT_TOKENS = 330  # the rubric and the request around it
REPLY_TOKENS = 30


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m birdcount.deploy.calibrate", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, text in (
        ("rate", "rate new pictures"),
        ("propose", "suggest thresholds"),
        ("run", "rate, then propose"),
    ):
        s = sub.add_parser(name, help=text)
        if name in ("rate", "run"):
            s.add_argument("--days", type=int, default=7, help="look back this many days of count log")
            s.add_argument(
                "--limit", type=int, default=5000, help="at most this many pictures per run (a week is ~4000)"
            )
            s.add_argument("--now", action="store_true", help="one request per picture instead of the Batch API")
            s.add_argument("--dry-run", action="store_true", help="only say what would be sent")
        if name in ("propose", "run"):
            s.add_argument("--window", type=int, default=14, help="days of ratings a proposal uses")
        s.add_argument("--model", help="vision model; default CALIBRATION_MODEL")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from birdcount.core.log import setup_logging
    from birdcount.deploy.alarm import load_config
    from birdcount.deploy.settings import DeploySettings

    settings = DeploySettings.load()
    setup_logging(settings.log_level)
    if settings.count_log_dir is None or not settings.count_log_dir.is_dir():
        print(f"error: no count log at {settings.count_log_dir} (ENABLE_COUNT_LOGGER=1 writes it)", file=sys.stderr)
        return 2
    config = load_config(settings.alarm_config_path)
    model = args.model or settings.calibration_model
    evidence = settings.alarm_output_dir or config.output_dir
    if args.command in ("rate", "run"):
        code = rate(settings, evidence, model, args)
        if code or args.command == "rate":
            return code
    return propose(settings, config, evidence, model, args.window)


def rate(settings, evidence: Path, model: str, args) -> int:
    from birdcount.deploy import countlog
    from birdcount.deploy.calibrate import ratings
    from birdcount.deploy.calibrate.rater import ClaudeRater, Picture, RaterError

    root = settings.count_log_dir
    days = countlog.days(root)[: args.days]
    cands = ratings.candidates(root, days, evidence)
    rated = {r["snapshot"] for d in days for r in ratings.current(ratings.read(root / d), model)}
    picked = ratings.select(cands, rated, limit=args.limit)
    alarms = sum(c.key.startswith(countlog.ALARM_PREFIX) for c in picked)
    tokens_in = len(picked) * (SNAPSHOT_TOKENS + PROMPT_TOKENS)
    print(
        f"{len(cands)} pictures in {len(days)} day(s), {len(rated)} already rated by {model}; "
        f"sending {len(picked)} ({alarms} alarm pictures), about {tokens_in / 1e6:.2f} M input tokens "
        f"and {len(picked) * REPLY_TOKENS / 1e6:.3f} M output tokens{' at Batch API prices' if not args.now else ''}."
    )
    if args.dry_run or not picked:
        return 0
    try:
        rater = ClaudeRater(settings.anthropic_api_key.get_secret_value(), model, batch=not args.now)
    except RaterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    from birdcount.deploy.calibrate.images import MaskShader

    shader = MaskShader(settings.mask_dir)
    pictures = [Picture(f"p{i}", shader.jpeg(c.file, c.mac)) for i, c in enumerate(picked)]
    started = time.monotonic()
    try:
        results = rater.rate(pictures, on_progress=lambda text: print(f"  {text}", flush=True))
    except RaterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    by_day: dict[str, list[dict]] = {}
    failed = []
    for i, c in enumerate(picked):
        result = results.get(f"p{i}", "no answer")
        if isinstance(result, str):
            failed.append(f"{c.key}: {result}")
            continue
        by_day.setdefault(c.day, []).append(ratings.row(c, model, *result))
    for day, rows in by_day.items():
        ratings.append(root / day, rows)
    print(f"Rated {sum(map(len, by_day.values()))} of {len(picked)} in {time.monotonic() - started:.0f} s.")
    for line in failed[:10]:
        print(f"  not rated: {line}")
    return 1 if failed and not by_day else 0


def propose(settings, config, evidence: Path, model: str, window: int) -> int:
    from birdcount.deploy import countlog
    from birdcount.deploy.alarm.camera_ids import mac_of
    from birdcount.deploy.calibrate import ratings
    from birdcount.deploy.calibrate.fit import propose as fit

    root = settings.count_log_dir
    days = countlog.days(root)[:window]
    points: dict[str, list[tuple[float, int]]] = {}
    alarm_levels: dict[str, list[int]] = {}
    pairs: dict[str, list[tuple[str, int]]] = {}
    for day in days:
        labels = _labels(root / day)
        for r in ratings.current(ratings.read(root / day), model):
            mac, level = r["mac"], int(r["level"])
            if r["snapshot"].startswith(countlog.ALARM_PREFIX):
                alarm_levels.setdefault(mac, []).append(level)
            else:
                points.setdefault(mac, []).append((float(r["count"]), level))
            if r["snapshot"] in labels:
                pairs.setdefault(mac, []).append((labels[r["snapshot"]], level))
    proposals = []
    for camera_id, cam in sorted(config.cameras.items()):
        mac = mac_of(camera_id)
        p = fit(mac, cam.threshold, points.get(mac, []), alarm_levels.get(mac, []), pairs.get(mac, []))
        proposals.append({"camera_id": camera_id, **p.__dict__})
    out = {
        "made_at": time.time(),
        "model": model,
        "days": days,
        "proposals": proposals,
    }
    folder = settings.calibration_dir
    (folder / "proposals").mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    (folder / "proposals" / f"{datetime.now():%Y-%m-%d_%H%M%S}.json").write_text(text, encoding="utf-8")
    tmp = folder / "latest.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(folder / "latest.json")
    print(f"{'camera':22} {'now':>5} {'suggest':>7} {'ratings':>7} {'dense':>5}  note")
    for p in proposals:
        sug = f"{p['suggested']:.0f}" if p["suggested"] is not None else "-"
        print(f"{p['camera_id']:22} {p['threshold']:>5.0f} {sug:>7} {p['ratings']:>7} {p['dense']:>5}  {p['note']}")
    print(f"-> {folder / 'latest.json'}")
    return 0


def _labels(day_dir: Path) -> dict[str, str]:
    import csv

    from birdcount.deploy import countlog

    path = day_dir / countlog.LABELS_FILE
    out: dict[str, str] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("label"):
                    out[row["snapshot"]] = row["label"]
                else:
                    out.pop(row.get("snapshot", ""), None)
    return out


if __name__ == "__main__":
    sys.exit(main())
