"""Hyperparameter sweeps: one training run per grid point, then a leaderboard.

    python -m birdcount.training.train.sweep --grid lr=1e-5,5e-5 --grid waux=0.5,1 -- --epochs 200
    python -m birdcount.training.train.sweep --summary outputs/train/sweep-20261026

Every combination of the `--grid` values runs as its own training process
(`python -m birdcount.training.train`, so GPU memory is returned between
runs), in OUTPUT_ROOT/train/sweep-<time>/<key=value,...>/. Arguments after
`--` go to every run. A combination whose folder already has a finished
summary.json is skipped, so an interrupted sweep resumes by re-running the
same command with `--sweep-dir`.

The leaderboard reads each run's summary.json (v2 parsed checkpoint file
names) and ranks by the score that picks best checkpoints, 2·RMSE + MAE.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from birdcount.core.settings import CoreSettings
from birdcount.training import child, runs
from birdcount.training.runs import finished, leaderboard
from birdcount.training.train import config


def parse_grid(items: list[str]) -> dict[str, list[str]]:
    """['lr=1e-5,5e-5', ...] -> {'lr': ['1e-5', '5e-5'], ...}, checked against the training flags."""
    known = {a.dest for a in config.build_parser()._actions}
    grid = {}
    for item in items:
        key, sep, values = item.partition("=")
        key = key.strip().removeprefix("--").replace("-", "_")
        if not sep or not values.strip():
            raise ValueError(f"--grid {item!r}: expected key=v1,v2,...")
        if key not in known or key in ("output_dir", "resume", "name"):
            raise ValueError(f"--grid {item!r}: {key} is not a training option that can be swept")
        grid[key] = [v.strip() for v in values.split(",") if v.strip()]
    return grid


def combinations(grid: dict[str, list[str]]) -> list[dict[str, str]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


def run_name(combo: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in combo.items())


def flags(combo: dict[str, str]) -> list[str]:
    out = []
    for key, value in combo.items():
        flag = "--" + key.replace("_", "-")
        if value.lower() in ("true", "false"):  # booleans are --x / --no-x
            out.append(flag if value.lower() == "true" else "--no-" + key.replace("_", "-"))
        else:
            out += [flag, *value.split()]
    return out


def print_leaderboard(rows: list[dict]) -> None:
    print(f"\n{'run':<40} {'status':<11} {'epoch':>5} {'MAE':>7} {'RMSE':>7} {'count acc':>9} {'Acc@5/10%':>9}")
    for r in rows:
        if r["score"] is None:
            print(f"{r['run'][:40]:<40} {r['status'] or '-':<11}   (no validation yet)")
            continue
        print(
            f"{r['run'][:40]:<40} {r['status']:<11} {r['epoch']:>5} {r['mae']:>7.2f} {r['rmse']:>7.2f} "
            f"{r['counting_accuracy'] * 100:>8.1f}% {r['primary_accuracy'] * 100:>8.1f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m birdcount.training.train.sweep",
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s [options] [-- training flags for every run]",
    )
    p.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (repeatable)")
    p.add_argument("--sweep-dir", type=Path, help="sweep folder (default: OUTPUT_ROOT/train/sweep-<time>)")
    p.add_argument("--summary", type=Path, metavar="SWEEP_DIR", help="only print the leaderboard of a sweep")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    return p


def run(cmd: list[str]) -> int:
    """One training run; Ctrl+C (or a stop from the web UI) lets it save its state first."""
    stop_file = Path(tempfile.gettempdir()) / f"birdcount-sweep-{os.getpid()}.stop"
    proc = subprocess.Popen(child.command(*cmd), env=child.env(stop_file))
    try:
        return child.wait(proc)
    except KeyboardInterrupt:
        child.stop(stop_file)
        child.wait(proc)
        raise
    finally:
        stop_file.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    passthrough = argv[argv.index("--") + 1 :] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    parser = build_parser()
    args = parser.parse_args(own)

    if args.summary is not None:
        print_leaderboard(leaderboard(args.summary))
        return 0
    try:
        grid = parse_grid(args.grid)
    except ValueError as e:
        parser.error(str(e))
    if not grid:
        parser.error("give at least one --grid key=v1,v2,...")
    sweep_dir = args.sweep_dir or runs.sweep_dir(CoreSettings.load().output_root)
    combos = combinations(grid)
    print(f"{len(combos)} run(s) in {sweep_dir}", flush=True)
    status = 0
    for i, combo in enumerate(combos, 1):
        run_dir = sweep_dir / run_name(combo)
        if finished(run_dir):
            print(f"[{i}/{len(combos)}] {run_dir.name}: done, skipped", flush=True)
            continue
        resume = ["--resume", str(run_dir)] if (run_dir / "last.tar").is_file() else None
        cmd = ["birdcount.training.train", *(resume or ["--output-dir", str(run_dir), *flags(combo), *passthrough])]
        print(f"[{i}/{len(combos)}] python -m {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        try:
            code = run(cmd)
        except KeyboardInterrupt:
            code = 130
        if code == 130:  # stopped in the run: stop the sweep too
            print("Interrupted.", flush=True)
            status = 130
            break
        if code:
            print(f"  exited with {code}; continuing with the next run", flush=True)
    if not args.dry_run:
        print_leaderboard(leaderboard(sweep_dir))
    return status


if __name__ == "__main__":
    sys.exit(main())
