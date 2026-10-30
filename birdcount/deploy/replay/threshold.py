"""Trying other alarm thresholds on a replay, without counting again.

    python -m birdcount.deploy.replay.threshold B8A44FD51C3C --try 100..140      # what each would have done
    python -m birdcount.deploy.replay.threshold B8A44FD51C3C --set 110           # the run's alarms at 110
    python -m birdcount.deploy.webui --replay outputs/replay/full                # check them

A camera-day's samples (deploy.replay.day) hold every count with its time and
centroid, so the alarm engine decides on them again at any threshold exactly
as it did while counting (10 s trigger, smoothing, hysteresis, motion
filter), in seconds instead of the hours the counting took.

--try prints, per threshold, the camera's Level 1 alarms over its days, scored
against the verdicts given so far on the farm backend's Check alarms page: an
alarm overlapping in time one judged a real pile-up counts as real, one
overlapping only false alarms as false. A verdict given at one threshold so
scores the alarms of every other: check the alarms of the lowest threshold
worth considering, and every higher one is scored.

--set makes T the camera's threshold in the run (thresholds.json, also for days
counted later): its alarm evidence with the pictures (only those frames are
decoded again, from the recordings), the count log's alarm levels and the
summary, as if counted at T. Each threshold keeps its own evidence and the
verdicts given on it. configs/alarm.json is never changed.

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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("birdcount.deploy.replay.threshold")

# Verdicts on the Check alarms page (webui.store.LABELS), and what an alarm overlapping them counts as.
VERDICTS = {"pileup": "real", "normal": "false", "unsure": "unsure"}


@dataclass(frozen=True)
class Episode:
    """One alarm event: from Level 1 to its recovery."""

    start: float
    end: float


def parse_thresholds(spec: str) -> list[float]:
    """'100..140' (every 10), '100..140:5', or '110,120,135'."""
    out: list[float] = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        span, _, step = part.partition(":")
        first, _, last = span.partition("..")
        a, b, s = float(first), float(last or first), float(step or 10)
        if b < a or s <= 0:
            raise ValueError(f"{part!r}: expected FROM..TO[:STEP] going up")
        n = round((b - a) / s)
        out += [a + i * s for i in range(n + 1)]
    return sorted(set(out))


# --- deciding again -----------------------------------------------------------------------------------


def decide(config, mac: str, samples) -> list:
    """The alarm engine's decision on every sample of a camera-day, as made while counting."""
    from birdcount.deploy.handlers.engine import AlarmEngine
    from birdcount.deploy.results import empty_batch

    rec = empty_batch(len(samples["t"]))
    rec["captured"] = rec["inferred"] = samples["t"]
    rec["count"], rec["cx"], rec["cy"] = samples["count"], samples["cx"], samples["cy"]
    rec["frame_idx"] = samples["frame"]
    decisions, _ = AlarmEngine(config, [mac]).evaluate(rec, set())
    return decisions


def episodes(decisions, t) -> list[Episode]:
    """Each alarm event, from its Level 1 to its recovery (or the day's last sample)."""
    out, start = [], None
    for d in decisions:
        for a in d.actions if d is not None else ():
            if a.action_type == "level1":
                start = a.timestamp
            elif a.action_type == "recovery" and start is not None:
                out.append(Episode(start, a.timestamp))
                start = None
    if start is not None:
        out.append(Episode(start, float(t[-1])))
    return out


def _try_one(args: tuple) -> tuple[float, dict[str, list[Episode]]]:
    """Episodes per day at one threshold (a process of its own: the engine is pure Python)."""
    import numpy as np

    from birdcount.deploy.alarm import load_config

    root, config_path, mac, threshold, days = args
    from birdcount.deploy.replay.layout import RunLayout

    layout = RunLayout(Path(root))
    config = load_config(config_path).with_thresholds({mac: threshold})
    logging.disable(logging.WARNING)  # every alarm of every threshold is not news
    out = {}
    for day in days:
        with np.load(layout.samples_of(mac, day)) as z:
            samples = {k: z[k] for k in ("t", "count", "cx", "cy", "frame")}
        out[day] = episodes(decide(config, mac, samples), samples["t"])
    return threshold, out


# --- verdicts -----------------------------------------------------------------------------------------


def verdicts(layout, mac: str) -> list[tuple[Episode, str]]:
    """Every alarm of this camera given a verdict, at whatever threshold, with its verdict."""
    from birdcount.deploy import countlog

    out = []
    for day_dir in sorted(p for p in layout.count_log.iterdir() if p.is_dir()) if layout.count_log.is_dir() else []:
        for key, row in countlog.read_labels(day_dir).items():
            if not key.startswith(countlog.ALARM_PREFIX):
                continue
            rel = Path(key.removeprefix(countlog.ALARM_PREFIX))
            if len(rel.parts) < 3 or rel.parts[1] != mac:
                continue
            event = _event_span(layout.evidence / rel.parent / "actions.jsonl")
            if event is not None:
                out.append((event, row["label"]))
    return out


def _event_span(actions: Path) -> Episode | None:
    try:
        stamps = [json.loads(line) for line in actions.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return None
    if not stamps:
        return None
    start = min(a["timestamp"] for a in stamps)
    ends = [a["timestamp"] for a in stamps if a["action_type"] == "recovery"]
    return Episode(start, max(ends) if ends else max(a["timestamp"] for a in stamps))


def score(episode: Episode, judged: list[tuple[Episode, str]]) -> str:
    """real / false / unsure from the verdicts on alarms overlapping it in time; unchecked without any."""
    found = {VERDICTS.get(label, "unsure") for e, label in judged if e.start <= episode.end and episode.start <= e.end}
    for verdict in ("real", "false", "unsure"):
        if verdict in found:
            return verdict
    return "unchecked"


# --- setting one --------------------------------------------------------------------------------------


def _set_day(args: tuple) -> dict:
    """One camera-day's alarm at a new threshold: evidence, count-log levels, samples, record."""
    import numpy as np

    from birdcount.core.video import frames_at
    from birdcount.deploy.alarm import load_config
    from birdcount.deploy.replay.day import AlarmFiling, alarm_numbers
    from birdcount.deploy.replay.layout import RunLayout

    root, config_path, mac, threshold, day, frame_size = args
    layout = RunLayout(Path(root))
    config = load_config(config_path).with_thresholds({mac: threshold})
    logging.disable(logging.WARNING)
    with np.load(layout.samples_of(mac, day)) as z:
        samples = {k: z[k] for k in z.files}
    decisions = decide(config, mac, samples)

    wanted: dict[int, list[int]] = {}  # clip -> samples whose picture an alarm action needs
    for i, d in enumerate(decisions):
        if d is not None and d.actions:
            wanted.setdefault(int(samples["clip"][i]), []).append(i)
    pictures: dict[int, object] = {}
    for clip, idx in wanted.items():
        path = Path(str(samples["clips"][clip]))
        try:
            got = frames_at(path, [int(samples["frame"][i]) for i in idx], tuple(frame_size))
        except OSError as e:
            got = {}
            logging.getLogger(__name__).error("No alarm pictures from %s: %s", path, e)
        for i in idx:
            pictures[i] = got.get(int(samples["frame"][i]))

    folder = layout.evidence_of(mac, day, threshold)
    shutil.rmtree(folder, ignore_errors=True)
    filing = AlarmFiling(folder, config)
    levels, excluded = filing.file(decisions, pictures.get)

    samples["level"], samples["excluded"] = levels, excluded
    tmp = layout.samples_of(mac, day).with_name(f"{day}.tmp.npz")
    np.savez_compressed(tmp, **samples)
    tmp.replace(layout.samples_of(mac, day))
    _relevel(layout.count_log / day / f"{mac}.csv", samples["t"], levels, excluded)

    record = layout.record(mac, day)
    record.update(alarm_numbers(samples["count"], threshold, record["fps"], filing.alarms))
    per_clip = np.zeros(len(record["clips"]), np.int8)
    np.maximum.at(per_clip, samples["clip"], levels)
    for entry, level in zip(record["clips"], per_clip.tolist(), strict=True):
        if entry["samples"]:
            entry["max_level"] = level
    return record


def _relevel(path: Path, t, levels, excluded) -> None:
    """The count log's alarm level and motion exclusions per second, from the samples' new ones."""
    import numpy as np

    from birdcount.deploy.countlog import ROW_FIELDS

    seconds, inverse = np.unique(t.astype(np.int64), return_inverse=True)
    level = np.zeros(len(seconds), np.int8)
    np.maximum.at(level, inverse, levels)
    out_of = np.bincount(inverse, weights=excluded.astype(np.float64), minlength=len(seconds))
    by_second = {int(s): (int(lv), int(x)) for s, lv, x in zip(seconds, level, out_of, strict=True)}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["alarm_level"], row["motion_excluded_frames"] = by_second.get(int(row["timestamp"]), (0, 0))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


# --- the command --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m birdcount.deploy.replay.threshold",
        description="Try other alarm thresholds on a replay without counting again.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("camera", help="the camera's MAC")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--try", dest="try_", metavar="T", help="thresholds to compare: 100..140, 100..140:5, 110,120")
    what.add_argument("--set", type=float, metavar="T", help="make T the camera's threshold in the run")
    p.add_argument("--run", default="full", help="run folder under OUTPUT_ROOT/replay/")
    p.add_argument("--alarm-config", type=Path, default=None, help="alarm rules; default the run's")
    p.add_argument("--jobs", type=int, default=8, help="processes")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mac = args.camera.strip().upper()

    from birdcount.core.log import setup_logging
    from birdcount.deploy.alarm import load_config
    from birdcount.deploy.replay.layout import RunLayout
    from birdcount.deploy.settings import DeploySettings

    settings = DeploySettings.load()
    setup_logging(settings.log_level)
    layout = RunLayout(settings.output_root / "replay" / args.run)
    run_file = layout.root / "run.json"
    if not run_file.is_file():
        parser.error(f"no replay in {layout.root}")
    about = json.loads(run_file.read_text(encoding="utf-8"))
    config_path = args.alarm_config or Path(about["alarm_config"])
    config = load_config(config_path)
    if config.threshold_of(mac) is None:
        parser.error(f"{mac} is not in the alarm config {config_path}: it has no alarm to try")
    days = [p.stem for p in sorted((layout.samples / mac).glob("*.json"))]
    if not days:
        logger.error("No camera-days of %s counted in %s.", mac, layout.root)
        return 2

    if args.try_:
        try:
            thresholds = parse_thresholds(args.try_)
        except ValueError as e:
            parser.error(f"--try {e}")
        _print_try(layout, mac, config, config_path, thresholds, days, args.jobs)
        return 0

    jobs = [(str(layout.root), str(config_path), mac, args.set, day, about["frame_size"]) for day in days]
    logger.info("%s at %g over %d day(s) of %s ...", mac, args.set, len(days), layout.root)
    with ProcessPoolExecutor(min(args.jobs, len(jobs))) as pool:
        records = list(pool.map(_set_day, jobs))
    layout.set_threshold(mac, args.set)
    for day, record in zip(days, records, strict=True):
        layout.finish(mac, day, record)
    level1 = sum(r["alarms"].get("level1", 0) for r in records)
    logger.info("%s now alarms at %g in %s: %d Level 1 alarm(s).", mac, args.set, layout.root, level1)
    logger.info("Check them: python -m birdcount.deploy.webui --replay %s", layout.root)
    return 0


def _print_try(layout, mac: str, config, config_path: Path, thresholds: list[float], days: list[str], jobs: int):
    judged = verdicts(layout, mac)
    work = [(str(layout.root), str(config_path), mac, t, days) for t in thresholds]
    with ProcessPoolExecutor(min(jobs, len(work))) as pool:
        results = sorted(pool.map(_try_one, work))
    in_run = layout.thresholds().get(mac, config.threshold_of(mac))
    hours = sum(layout.record(mac, d)["hours"] for d in days)
    print(
        f"{mac}: {len(days)} day(s), {hours:.1f} h; threshold in alarm.json {config.threshold_of(mac):g}, in this run {in_run:g}"
    )
    print(f"verdicts given on {len(judged)} alarm(s) of this camera so far\n")
    print(f"{'T':>6} {'Level 1':>8} {'per day':>8} {'alarming':>9} {'days':>5}   real  false  unsure  unchecked")
    for t, per_day in results:
        eps = [e for day in days for e in per_day[day]]
        tally = dict.fromkeys(("real", "false", "unsure", "unchecked"), 0)
        for e in eps:
            tally[score(e, judged)] += 1
        minutes = sum(e.end - e.start for e in eps) / 60
        print(
            f"{t:>6g} {len(eps):>8} {len(eps) / max(hours / 13, 1e-9):>8.1f} {minutes:>7.1f} m {sum(1 for d in days if per_day[d]):>5}"
            f"   {tally['real']:>4}  {tally['false']:>5}  {tally['unsure']:>6}  {tally['unchecked']:>9}"
        )
    print("\nper day = per 13 h of recording. Verdicts come from the Check alarms page of any threshold tried.")


if __name__ == "__main__":
    sys.exit(main())
