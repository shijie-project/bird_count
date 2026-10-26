"""Where each training tool writes its runs, and how to find them again.

    OUTPUT_ROOT/train/<time>[-name]/                python -m birdcount.training.train
    OUTPUT_ROOT/train/sweep-<time>/<combination>/   python -m birdcount.training.train.sweep
    OUTPUT_ROOT/eval/<checkpoint>/<split>-<time>/   python -m birdcount.training.eval
    OUTPUT_ROOT/video/<video>/<time>/               python -m birdcount.training.video

The tools create these folders; the web UI lists them, so a run started from a
shell shows up next to the ones started from the UI. A run passed an explicit
--output-dir is wherever it was told to go.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


STAMP_FORMAT = "%Y%m%d-%H%M%S"
_STAMP_RE = re.compile(r"(\d{8}-\d{6})")

KINDS = ("train", "sweep", "eval", "video")


def stamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime(STAMP_FORMAT)


def train_dir(output_root: Path, name: str = "", now: datetime | None = None) -> Path:
    return output_root / "train" / (stamp(now) + (f"-{name}" if name else ""))


def sweep_dir(output_root: Path, now: datetime | None = None) -> Path:
    return output_root / "train" / f"sweep-{stamp(now)}"


def eval_dir(output_root: Path, checkpoint: str | Path, split: str, now: datetime | None = None) -> Path:
    return output_root / "eval" / Path(checkpoint).stem / f"{split}-{stamp(now)}"


def video_dir(output_root: Path, video: Path, run_stamp: str) -> Path:
    """One video's folder; every video of one invocation shares `run_stamp`."""
    return output_root / "video" / video.stem / run_stamp


# --- reading runs back -------------------------------------------------------------------


def score(mae: float, rmse: float) -> float:
    """Lower is better. RMSE weighs the large misses (pile-ups) the alarm depends on."""
    return 2.0 * rmse + mae


def read_json(path: Path) -> dict | None:
    """The JSON object in `path`, or None while it is missing or half-written."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def finished(run_dir: Path) -> bool:
    """A training run that ended by itself (an interrupted one can be resumed)."""
    summary = read_json(run_dir / "summary.json")
    return summary is not None and summary.get("status") != "interrupted"


def leaderboard(sweep: Path) -> list[dict]:
    """The sweep's runs, best first by `score`; runs without a validation last."""
    rows = []
    for summary_path in sorted(sweep.glob("*/summary.json")):
        data = read_json(summary_path) or {}
        best = data.get("best") or {}
        rows.append(
            {
                "run": summary_path.parent.name,
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


def kind_of(run_dir: Path) -> str | None:
    """Which tool wrote `run_dir`, from the files it holds (None: not a run folder)."""
    if (run_dir / "config.json").is_file():
        return "train"
    if run_dir.name.startswith("sweep-") and run_dir.parent.name == "train":
        return "sweep"
    if (run_dir / "samples.jsonl").is_file() or (run_dir / "summary.json").is_file():
        return "video"
    if (
        (run_dir / "metrics.json").is_file()
        or (run_dir / "per_image.csv").is_file()
        or (run_dir / "overlays").is_dir()
    ):
        return "eval"
    return None


def find(output_root: Path, kind: str) -> list[Path]:
    """Run folders of `kind` in the default layout, newest first."""
    if kind == "train":
        found = [
            p.parent for p in (output_root / "train").glob("*/config.json") if not p.parent.name.startswith("sweep-")
        ]
    elif kind == "sweep":
        found = [p for p in (output_root / "train").glob("sweep-*") if p.is_dir()]
    elif kind == "eval":
        found = [p for p in (output_root / "eval").glob("*/*") if p.is_dir()]
    elif kind == "video":
        found = [p.parent for p in (output_root / "video").glob("*/*/summary.json")]
    else:
        raise ValueError(f"unknown run kind {kind!r}")
    return sorted(found, key=started, reverse=True)


def under(folder: Path) -> list[Path]:
    """The runs in `folder`: itself if it is one, else the runs directly inside it
    (a video tool given several videos writes one folder per video)."""
    if kind_of(folder) is not None:
        return [folder]
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_dir() and kind_of(p) is not None), key=lambda p: p.name)


def started(run_dir: Path) -> float:
    """When the run started: the stamp in its folder name, else the folder's mtime."""
    match = _STAMP_RE.search(run_dir.name)
    if match:
        try:
            return datetime.strptime(match[1], STAMP_FORMAT).timestamp()
        except ValueError:
            pass
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return 0.0
