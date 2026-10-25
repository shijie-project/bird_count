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
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from birdcount.core.settings import CoreSettings
from birdcount.training.train.config import build_parser
from birdcount.training.train.trainer import score


def parse_grid(items: list[str]) -> dict[str, list[str]]:
    """['lr=1e-5,5e-5', ...] -> {'lr': ['1e-5', '5e-5'], ...}, checked against the training flags."""
    known = {a.dest for a in build_parser()._actions}
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


def finished(run_dir: Path) -> bool:
    summary = run_dir / "summary.json"
    return summary.is_file() and json.loads(summary.read_text(encoding="utf-8")).get("status") != "interrupted"


def leaderboard(sweep_dir: Path) -> list[dict]:
    rows = []
    for summary in sorted(sweep_dir.glob("*/summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        best = data.get("best") or {}
        rows.append(
            {
                "run": summary.parent.name,
                "status": data.get("status"),
                "epoch": best.get("epoch"),
                "mae": best.get("mae"),
                "rmse": best.get("rmse"),
                "counting_accuracy": best.get("counting_accuracy"),
                "primary_accuracy": best.get("primary_accuracy"),
                "score": score(best["mae"], best["rmse"]) if best else None,
            }
        )
    return sorted(rows, key=lambda r: (r["score"] is None, r["score"] or 0.0))


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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    passthrough = argv[argv.index("--") + 1 :] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(
        prog="python -m birdcount.training.train.sweep", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (repeatable)")
    parser.add_argument("--sweep-dir", type=Path, help="sweep folder (default: OUTPUT_ROOT/train/sweep-<time>)")
    parser.add_argument("--summary", type=Path, metavar="SWEEP_DIR", help="only print the leaderboard of a sweep")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
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
    sweep_dir = args.sweep_dir or CoreSettings.load().output_root / "train" / f"sweep-{datetime.now():%Y%m%d-%H%M%S}"
    combos = combinations(grid)
    print(f"{len(combos)} run(s) in {sweep_dir}")
    for i, combo in enumerate(combos, 1):
        run_dir = sweep_dir / run_name(combo)
        if finished(run_dir):
            print(f"[{i}/{len(combos)}] {run_dir.name}: done, skipped")
            continue
        resume = ["--resume", str(run_dir)] if (run_dir / "last.tar").is_file() else None
        cmd = [sys.executable, "-m", "birdcount.training.train"]
        cmd += resume or ["--output-dir", str(run_dir), *flags(combo), *passthrough]
        print(f"[{i}/{len(combos)}] {' '.join(cmd)}")
        if not args.dry_run:
            code = subprocess.call(cmd)
            if code == 130:  # Ctrl+C in the run: stop the sweep too
                print("Interrupted.")
                break
            if code:
                print(f"  exited with {code}; continuing with the next run")
    if not args.dry_run:
        print_leaderboard(leaderboard(sweep_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
