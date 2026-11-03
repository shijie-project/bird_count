"""Trying other alarm thresholds on the farm's own count log, scored by the verdicts given on its pictures.

    python -m birdcount.deploy.calibrate try B8A44FD51C3C 100..140
    python -m birdcount.deploy.calibrate try B8A44FD51C3C 100..140:5 --days 30

The count log keeps every camera's count each second, so the alarm engine
decides again on those seconds at any threshold (10 s trigger, smoothing,
hysteresis, motion filter all run in seconds): what each threshold would have
alarmed on. Decided on per-second means rather than every frame, so at the
threshold in force it comes close to, not exactly, the alarms the runtime
raised; the first line says how close.

Each alarm is then scored against the verdicts given on the farm backend:
on the pictures around the threshold (the count log keeps one every 30 s
while a camera is near it) and on the alarm pictures. An alarm holding a
picture judged a pile-up is real, one holding only pictures judged normal is
false. A pile-up judged on a picture no alarm covers is one that threshold
misses. The same convention as `deploy.replay.threshold --try` on a replay.
"""

from __future__ import annotations

import csv
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from birdcount.deploy import countlog
from birdcount.deploy.replay.threshold import VERDICTS, Episode, decide, episodes, event_span, overlaps, score


@dataclass(frozen=True)
class Day:
    """One camera-day of the count log as the engine sees it: a sample a second."""

    day: str
    t: list[float]
    count: list[float]
    cx: list[float]
    cy: list[float]
    level: list[int]  # the alarm level the runtime logged

    @property
    def hours(self) -> float:
        return len(self.t) / 3600


def read_day(root: Path, day: str, mac: str) -> Day | None:
    path = root / day / f"{mac}.csv"
    if not path.is_file():
        return None
    t, count, cx, cy, level = [], [], [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t.append(int(row["timestamp"]) + 0.5)  # the middle of the second its frames came in
                count.append(float(row["count_mean"]))
                cx.append(float(row["centroid_x"]) if row["centroid_x"] else math.nan)
                cy.append(float(row["centroid_y"]) if row["centroid_y"] else math.nan)
                level.append(int(row["alarm_level"] or 0))
            except (KeyError, ValueError):
                continue
    return Day(day, t, count, cx, cy, level) if t else None


def alarms_at(config, mac: str, threshold: float, day: Day) -> list[Episode]:
    import numpy as np

    samples = {
        "t": np.asarray(day.t, np.float64),
        "count": np.asarray(day.count, np.float32),
        "cx": np.asarray(day.cx, np.float32),
        "cy": np.asarray(day.cy, np.float32),
        "frame": np.zeros(len(day.t), np.int64),
    }
    return episodes(decide(config.with_thresholds({mac: threshold}), mac, samples), samples["t"])


def logged_alarms(day: Day) -> int:
    """Level 1 alarms the runtime raised that day: each rise of the logged level from 0."""
    return sum(1 for a, b in zip([0, *day.level], day.level, strict=False) if a == 0 and b > 0)


def verdicts(root: Path, evidence: Path, mac: str, days: list[str]) -> list[tuple[Episode, str]]:
    """Every verdict on this camera's pictures: a snapshot as the moment it shows, an alarm picture as its alarm."""
    out = []
    for day in days:
        for key, row in countlog.read_labels(root / day).items():
            if key.startswith(countlog.ALARM_PREFIX):
                rel = Path(key.removeprefix(countlog.ALARM_PREFIX))
                if mac not in rel.as_posix():
                    continue
                span = event_span(evidence / rel.parent / "actions.jsonl")
                if span is not None:
                    out.append((span, row["label"]))
            else:
                info = countlog.parse_snapshot(day, key)
                if info is not None and info["mac"] == mac:
                    out.append((Episode(info["t"], info["t"] + 1), row["label"]))
    return out


def _try_one(args: tuple) -> tuple[float, dict[str, list[Episode]]]:
    """Alarms per day at one threshold (a process of its own: the engine is pure Python)."""
    from birdcount.deploy.alarm import load_config

    root, config_path, mac, threshold, days = args
    config = load_config(config_path)
    logging.disable(logging.WARNING)  # every alarm of every threshold is not news
    out = {}
    for name in days:
        day = read_day(Path(root), name, mac)
        out[name] = alarms_at(config, mac, threshold, day) if day else []
    return threshold, out


def run(root: Path, evidence: Path, config_path: Path, mac: str, thresholds: list[float], days: list[str], jobs: int):
    from birdcount.deploy.alarm import load_config

    config = load_config(config_path)
    now = config.threshold_of(mac)
    if now is None:
        print(f"error: {mac} has no threshold in {config_path}")
        return 2
    loaded = {d: day for d in days if (day := read_day(root, d, mac)) is not None}
    days = sorted(loaded)
    if not days:
        print(f"error: no count log of {mac} in {root}")
        return 2
    hours = sum(d.hours for d in loaded.values())
    judged = verdicts(root, evidence, mac, days)
    piles = [e for e, label in judged if VERDICTS.get(label) == "real"]
    work = [(str(root), str(config_path), mac, t, days) for t in sorted({*thresholds, now})]
    with ProcessPoolExecutor(max(1, min(jobs, len(work)))) as pool:
        results = dict(pool.map(_try_one, work))
    again = sum(len(results[now][d]) for d in days)
    logged = sum(logged_alarms(loaded[d]) for d in days)
    print(f"{mac}: {len(days)} day(s) {days[0]}..{days[-1]}, {hours:.1f} h of count log; threshold now {now:g}")
    print(f"at {now:g}, decided again on per-second counts: {again} Level 1 alarm(s); the runtime raised {logged}")
    print(f"verdicts so far: {len(judged)} picture(s), {len(piles)} judged a pile-up")
    print()
    print(f"{'T':>6} {'Level 1':>8} {'per day':>8} {'alarming':>9}   real  false  unsure  unchecked   pile-ups missed")
    for t in sorted(results):
        eps = [e for d in days for e in results[t][d]]
        tally = dict.fromkeys(("real", "false", "unsure", "unchecked"), 0)
        for e in eps:
            tally[score(e, judged)] += 1
        missed = sum(1 for p in piles if not any(overlaps(p, e) for e in eps))
        minutes = sum(e.end - e.start for e in eps) / 60
        mark = " <- now" if t == now else ""
        print(
            f"{t:>6g} {len(eps):>8} {len(eps) / max(hours / 14, 1e-9):>8.1f} {minutes:>7.1f} m"
            f"   {tally['real']:>4}  {tally['false']:>5}  {tally['unsure']:>6}  {tally['unchecked']:>9}"
            f"   {missed:>15}{mark}"
        )
    print("\nper day = per 14 h of count log. Judge the pictures on the farm backend to fill real / false / missed.")
    return 0
