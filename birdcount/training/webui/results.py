"""What the page shows about a run, read from the files its tool wrote.

    train   config.json, metrics.jsonl (one line per epoch), summary.json, train.log
    sweep   the summary.json of each run inside it (the leaderboard)
    eval    metrics.json, per_image.csv, blobs.json / grid.json, overlays/
    video   summary.json, samples.jsonl (one line per sample)

The line files are read from a byte offset the page passes back, so a poll
of a live run costs only the lines written since the last one. Nothing here
parses console output.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

from birdcount.training import runs


ACTIVE_SECONDS = 180  # a run folder written to this recently, with no summary, is still going
WORST_REGIONS = 200  # region rows sent per eval run
LOG_TAIL = 64 * 1024  # bytes of a log sent on the first read


def read_lines(path: Path, offset: int, limit: int = 32 * 2**20) -> tuple[list[str], int]:
    """Complete lines of `path` from byte `offset`, and the offset after them."""
    try:
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read(limit)
    except OSError:
        return [], offset
    end = data.rfind(b"\n") + 1  # a line still being written waits for the next read
    lines = data[:end].decode("utf-8", errors="replace").splitlines()
    return [line for line in lines if line.strip()], offset + end


def read_jsonl(path: Path, offset: int) -> tuple[list[dict], int]:
    lines, offset = read_lines(path, offset)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records, offset


def read_log(path: Path, offset: int) -> dict:
    """Text of a log from `offset`; a negative offset starts at its last LOG_TAIL bytes."""
    try:
        size = path.stat().st_size
    except OSError:
        return {"text": "", "offset": 0, "missing": True}
    if offset < 0 or offset > size:  # first read, or the file was rewritten
        offset = max(0, size - LOG_TAIL)
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(size - offset)
    return {"text": data.decode("utf-8", errors="replace"), "offset": offset + len(data)}


def last_line(path: Path) -> dict | None:
    """The last JSON line of `path`, read from its end."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def _recent(*paths: Path) -> bool:
    now = time.time()
    for path in paths:
        try:
            if now - path.stat().st_mtime < ACTIVE_SECONDS:
                return True
        except OSError:
            continue
    return False


def _status(done: str | None, live: bool, *touched: Path) -> str:
    """done: how the run ended by its own files (None: no end written yet)."""
    if live:
        return "running"
    if done is not None:
        return {"finished": "done", "early_stop": "done", "interrupted": "stopped"}.get(done, done)
    return "running" if _recent(*touched) else "incomplete"


# --- one line per run, for the lists -------------------------------------------------------------


def overview(run_dir: Path, kind: str, live: bool) -> dict:
    """Name, status and headline numbers of a run, cheap enough for a list of hundreds."""
    out = {"kind": kind, "name": run_dir.name, "started": runs.started(run_dir)}
    if kind == "train":
        summary = runs.read_json(run_dir / "summary.json")
        config = runs.read_json(run_dir / "config.json") or {}
        last = last_line(run_dir / "metrics.jsonl")
        out["status"] = _status(
            summary and summary.get("status"), live, run_dir / "metrics.jsonl", run_dir / "train.log"
        )
        out["epochs"] = [(last or {}).get("epoch", -1) + 1, config.get("epochs")]
        best = (summary or {}).get("best") or _best_so_far(run_dir)
        if best:
            out["best"] = {k: best.get(k) for k in ("epoch", "mae", "rmse", "counting_accuracy", "primary_accuracy")}
    elif kind == "sweep":
        rows = runs.leaderboard(run_dir)
        statuses = {r["status"] for r in rows}
        if live:
            out["status"] = "running"
        elif "interrupted" in statuses:
            out["status"] = "stopped"
        else:
            out["status"] = "done" if rows and None not in statuses else "incomplete"
        out["runs"] = len(rows)
        if rows and rows[0]["score"] is not None:
            out["best"] = rows[0]
    elif kind == "eval":
        metrics = runs.read_json(run_dir / "metrics.json")
        out["name"] = f"{run_dir.parent.name}/{run_dir.name}"
        out["status"] = _status("finished" if metrics else None, live, run_dir, run_dir / "overlays")
        if metrics:
            overall = metrics.get("overall", {})
            out["split"] = metrics.get("run", {}).get("split")
            out["images"] = metrics.get("run", {}).get("images")
            out["counting_accuracy"] = overall.get("counting_accuracy")
            out["primary_accuracy"] = metrics.get("tolerance_accuracy", {}).get("primary", {}).get("accuracy")
            out["mae"] = overall.get("mae")
        else:
            out["split"] = run_dir.name.rsplit("-", 2)[0]
    elif kind == "video":
        summary = runs.read_json(run_dir / "summary.json") or {}
        out["name"] = f"{run_dir.parent.name}/{run_dir.name}"
        done = summary.get("status") if summary.get("status") != "running" else None
        out["status"] = _status(done, live, run_dir / "samples.jsonl")
        out["video"] = summary.get("video", {}).get("name")
        if "summary" in summary:
            out["max_count"] = summary["summary"]["count"]["max"]
            out["max_peak"] = summary["summary"]["peak"]["max"]
    return out


# Files whose change can change a run's overview, by kind.
_WATCHED = {
    "train": ("summary.json", "metrics.jsonl", "config.json"),
    "eval": ("metrics.json",),
    "video": ("summary.json",),
}
_overviews: dict[tuple[Path, str], tuple[tuple, dict]] = {}
_overviews_lock = threading.Lock()


def _signature(run_dir: Path, kind: str) -> tuple:
    paths = [run_dir, *(run_dir / name for name in _WATCHED.get(kind, ()))]
    if kind == "sweep":
        paths += sorted(run_dir.glob("*/summary.json"))
    stamps = []
    for path in paths:
        try:
            st = path.stat()
            stamps.append((path.name, st.st_mtime_ns, st.st_size))
        except OSError:
            stamps.append((path.name, None, None))
    return tuple(stamps)


def overview_cached(run_dir: Path, kind: str, live: bool) -> dict:
    """`overview`, kept for runs that have ended until their files change: a run list
    of hundreds of finished runs then costs a few stat calls each."""
    if live:
        return overview(run_dir, kind, live)
    key, sig = (run_dir, kind), _signature(run_dir, kind)
    with _overviews_lock:
        hit = _overviews.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    info = overview(run_dir, kind, live)
    if info["status"] not in ("running", "incomplete"):  # these depend on the clock, not just the files
        with _overviews_lock:
            _overviews[key] = (sig, info)
    return info


def _best_so_far(run_dir: Path) -> dict | None:
    """Best validation of a run still going, from the kept checkpoint names."""
    found = []
    for path in run_dir.glob("best_ep*_mae*_rmse*.pth"):
        try:
            epoch, mae, rmse = path.stem.removeprefix("best_ep").replace("_mae", " ").replace("_rmse", " ").split()
            found.append({"epoch": int(epoch), "mae": float(mae), "rmse": float(rmse)})
        except ValueError:
            continue
    return min(found, key=lambda b: runs.score(b["mae"], b["rmse"])) if found else None


# --- everything about one run ----------------------------------------------------------------------


def detail(run_dir: Path, kind: str, live: bool, offset: int = 0) -> dict:
    """The run's results; `offset` is where the previous read of its line file ended."""
    out = overview(run_dir, kind, live)
    out["path"] = str(run_dir)
    if kind == "train":
        out["config"] = runs.read_json(run_dir / "config.json")
        out["summary"] = runs.read_json(run_dir / "summary.json")
        out["records"], out["offset"] = read_jsonl(run_dir / "metrics.jsonl", offset)
        out["checkpoints"] = sorted(p.name for p in run_dir.glob("*.pth"))
        out["log"] = "train.log"
    elif kind == "sweep":
        out["leaderboard"] = runs.leaderboard(run_dir)
    elif kind == "eval":
        out.update(_eval(run_dir))
    elif kind == "video":
        out["summary"] = runs.read_json(run_dir / "summary.json")
        records, out["offset"] = read_jsonl(run_dir / "samples.jsonl", offset)
        out["samples"] = [[s["i"], s["t"], s["count"], s["peak"]] for s in records]
    return out


def _eval(run_dir: Path) -> dict:
    out: dict = {"metrics": runs.read_json(run_dir / "metrics.json"), "images": [], "regions": {}}
    try:
        with (run_dir / "per_image.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out["images"].append(
                    {
                        "name": row["name"],
                        "gt": float(row["gt"]),
                        "pred": float(row["pred"]),
                        "err": float(row["error"]),
                        "rel": float(row["rel_error_pct"]) if row["rel_error_pct"] else None,
                        "masked": row["masked"] == "1",
                    }
                )
    except (OSError, KeyError, ValueError):
        pass
    for key in ("blobs", "grid"):
        data = runs.read_json(run_dir / f"{key}.json")
        if not data:
            continue
        rows = [
            {"image": im["name"], **{k: r.get(k) for k in ("id", "row", "col", "gt", "count", "error")}}
            for im in data["images"]
            for r in im["regions"]
            if r.get("error") is not None
        ]
        rows.sort(key=lambda r: abs(r["error"]), reverse=True)
        out["regions"][key] = {"total": len(rows), "worst": rows[:WORST_REGIONS]}
    overlays = run_dir / "overlays"
    if overlays.is_dir():
        kinds: dict[str, list[str]] = {}
        for png in overlays.glob("*.png"):
            stem, _, suffix = png.stem.rpartition("_")
            kinds.setdefault(suffix, []).append(stem)
        out["overlays"] = {"dir": str(overlays), "kinds": sorted(kinds)}
    return out
