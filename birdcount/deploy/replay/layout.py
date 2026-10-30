"""A replay run's folder: where everything goes, and the indexes kept over it.

    OUTPUT_ROOT/replay/<run>/
        run.json                    what counts it: model + sha256, frame size, FPS
        thresholds.json             thresholds tried in place of the alarm config's ({MAC: T}, deploy.replay.threshold)
        count_log/<day>/            the runtime's count log (deploy.countlog): <MAC>.csv,
                                    snapshots/<MAC>/, and meta.json over every camera of the day
        alarm/events.jsonl          every alarm action of the run in time order (the farm backend's index)
        alarm/<day>/<MAC>/T<T>/     one camera-day's evidence at threshold T: events/, snapshots/,
                                    its own events.jsonl (each threshold tried keeps its own, and
                                    the verdicts given on its alarms)
        samples/<MAC>/<day>.npz     every sample as counted (deploy.replay.day)
        samples/<MAC>/<day>.json    the camera-day's summary, clips and meta; written last, so it
                                    marks the camera-day done
        summary.csv                 one row per camera-day done

The count log and the evidence are laid out as the runtime writes them, so
the farm backend reads a run as it reads the farm:

    python -m birdcount.deploy.webui --replay outputs/replay/<run>

Camera-days run in parallel, and each writes only its own files. What is
shared (a day's meta.json, alarm/events.jsonl, summary.csv) is rebuilt from
the done camera-days whenever one finishes, so an interrupted or repeated
camera-day never leaves them half-merged. A camera-day's record names the
threshold its alarms were decided at, and the index shows that threshold's evidence.

stdlib only: the farm backend imports it.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


# run.json keys that must match for two camera-days' counts to be comparable.
COMPARABLE = ("model_sha256", "frame_size", "fps")
SUMMARY_FIELDS = [
    "mac",
    "day",
    "clips",
    "failed_clips",
    "hours",
    "count_mean",
    "count_p95",
    "count_max",
    "threshold",
    "seconds_over_60pct",
    "seconds_over_80pct",
    "seconds_over_threshold",
    "level1",
    "level2",
    "level3",
    "counted_in_s",
]
_REPLACE_TRIES = 20  # os.replace onto a file a reader has open fails on Windows for that moment


class RunMismatch(ValueError):
    """The run was counted under another model, frame size or FPS."""


@dataclass(frozen=True)
class RunLayout:
    root: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    @property
    def count_log(self) -> Path:
        return self.root / "count_log"

    @property
    def evidence(self) -> Path:
        return self.root / "alarm"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    def evidence_of(self, mac: str, day: str, threshold: float | None) -> Path:
        return self.evidence / day / mac / ("T-" if threshold is None else f"T{threshold:g}")

    def samples_of(self, mac: str, day: str) -> Path:
        return self.samples / mac / f"{day}.npz"

    def record_of(self, mac: str, day: str) -> Path:
        return self.samples / mac / f"{day}.json"

    def done(self, mac: str, day: str) -> bool:
        return self.record_of(mac, day).is_file()

    def record(self, mac: str, day: str) -> dict:
        return json.loads(self.record_of(mac, day).read_text(encoding="utf-8"))

    def thresholds(self) -> dict[str, float]:
        """{MAC: threshold} tried in place of the alarm config's."""
        path = self.root / "thresholds.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def set_threshold(self, mac: str, threshold: float) -> None:
        with self._lock:
            _write_atomic(self.root / "thresholds.json", json.dumps({**self.thresholds(), mac: threshold}, indent=2))

    # --- the run ------------------------------------------------------------------------------

    def open(self, about: dict) -> None:
        """Start the run, or continue it if it was counted the same way; RunMismatch otherwise."""
        path = self.root / "run.json"
        if path.is_file():
            before = json.loads(path.read_text(encoding="utf-8"))
            differ = [k for k in COMPARABLE if before.get(k) != about.get(k)]
            if differ:
                detail = ", ".join(f"{k} {before.get(k)!r} -> {about.get(k)!r}" for k in differ)
                raise RunMismatch(f"{self.root} was counted differently ({detail}); start another run")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, json.dumps({**about, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    # --- one camera-day -------------------------------------------------------------------------

    def clear(self, mac: str, day: str) -> None:
        """Remove what an earlier or interrupted pass of this camera-day left."""
        self.record_of(mac, day).unlink(missing_ok=True)
        self.samples_of(mac, day).unlink(missing_ok=True)
        (self.count_log / day / f"{mac}.csv").unlink(missing_ok=True)
        shutil.rmtree(self.count_log / day / "snapshots" / mac, ignore_errors=True)
        shutil.rmtree(self.evidence / day / mac, ignore_errors=True)

    def finish(self, mac: str, day: str, record: dict) -> None:
        """Mark a camera-day done (or its record changed) and fold it into the run's shared files.

        `record` is the camera-day's summary. A new one takes in the meta.json
        its count logger just wrote (this camera only), for the day's merged one.
        """
        if "meta" not in record:
            meta = self.count_log / day / "meta.json"
            record = {**record, "meta": json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}}
        with self._lock:
            _write_atomic(self.record_of(mac, day), json.dumps(record, indent=2, ensure_ascii=False))
            self._merge_meta(day)
            self._index_events()
            self._write_summary()

    def records(self, day: str | None = None) -> list[dict]:
        """The done camera-days (of one day), by camera then day."""
        pattern = f"*/{day}.json" if day else "*/*.json"
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.samples.glob(pattern))]

    # --- what is shared, rebuilt ----------------------------------------------------------------

    def _merge_meta(self, day: str) -> None:
        """The day's meta.json over every camera replayed into it (each logger wrote only its own)."""
        records = self.records(day)
        metas = [r["meta"] for r in records if r.get("meta")]
        if not metas:
            return
        merged = {k: v for k, v in metas[-1].items() if k != "streams"}  # sid 0 is every camera here
        merged["snapshot_floors"] = {mac: v for m in metas for mac, v in m.get("snapshot_floors", {}).items()}
        # The threshold each camera's alarms were decided at: the farm backend draws it.
        merged["thresholds"] = {r["mac"]: r["threshold"] for r in records if r.get("threshold") is not None}
        merged["replayed"] = sorted(r["mac"] for r in records)
        _write_atomic(self.count_log / day / "meta.json", json.dumps(merged, indent=2, ensure_ascii=False))

    def _index_events(self) -> None:
        lines = []
        for r in self.records():
            path = self.evidence_of(r["mac"], r["day"], r.get("threshold")) / "events.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    lines.append((json.loads(line)["timestamp"], line))
                except (ValueError, KeyError):
                    continue
        lines.sort(key=lambda x: x[0])
        self.evidence.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.evidence / "events.jsonl", "".join(f"{line}\n" for _, line in lines))

    def _write_summary(self) -> None:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_row(r) for r in self.records())
        _write_atomic(self.root / "summary.csv", buf.getvalue())


def summary_row(record: dict) -> dict:
    c, over, levels = record["count"], record["seconds_over"], record["alarms"]
    return {
        "mac": record["mac"],
        "day": record["day"],
        "clips": len(record["clips"]),
        "failed_clips": sum(1 for clip in record["clips"] if clip.get("error")),
        "hours": f"{record['hours']:.2f}",
        "count_mean": c["mean"],
        "count_p95": c["p95"],
        "count_max": c["max"],
        "threshold": record["threshold"],
        "seconds_over_60pct": over.get("0.6", ""),
        "seconds_over_80pct": over.get("0.8", ""),
        "seconds_over_threshold": over.get("1.0", ""),
        "level1": levels.get("level1", 0),
        "level2": levels.get("level2", 0),
        "level3": levels.get("level3", 0),
        "counted_in_s": record["counted_in_s"],
    }


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(_REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(0.05)
