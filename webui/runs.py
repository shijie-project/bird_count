"""Subprocess supervision for web-UI runs: launch, stream, parse, stop.

One `Run` wraps one `python train.py ...` / `python test.py ...` process. A
reader thread drains the merged stdout+stderr into an in-memory line buffer
(also tee'd to a log file under logs/webui/) and feeds each line to a parser
that turns the scripts' own log format into structured metrics for the charts.

Nothing here imports torch — the heavy lifting stays in the child process, so
the server stays responsive and a crashed run can never take the UI down.
"""

import itertools
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .schema import ENTRYPOINTS


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "webui"

# Keep memory bounded on very long trainings; the full log always lives on disk.
MAX_LINES = 50_000

_IS_WINDOWS = os.name == "nt"


# --- log-line parsers -------------------------------------------------------
# These mirror trainer.py's `_format_train_log` / `val_epoch` and test.py's
# summary printers. A format change there degrades the charts, never the logs.

_RE_TRAIN = re.compile(
    r"Epoch (?P<epoch>\d+) Train \| lr (?P<lr>\S+) \| loss (?P<loss>\S+) .*?"
    r"\| MSE (?P<mse>\S+) \| MAE (?P<mae>\S+)"
)
_RE_VAL = re.compile(r"Epoch (?P<epoch>\d+) Val \((?P<tag>[^)]*)\) \| MSE (?P<mse>\S+) \| MAE (?P<mae>\S+)")
_RE_BEST = re.compile(r"saved best -> (?P<name>\S+)")
_RE_IMAGE = re.compile(
    r"^\s{2}(?P<name>.+?): GT\s+(?P<gt>-?[\d.]+)\s+Pred\s+(?P<pred>-?[\d.]+)\s+Err\s+(?P<err>\S+)\s+Err%\s+(?P<rel>\S+)\s*$"
)
_RE_OVERLAY_DIR = re.compile(r"Writing density overlays to:\s*(?P<path>.+?)\s*$")
_RE_SUMMARY_KV = re.compile(r"^\s{2}(?P<key>[A-Za-z][^:]*?)\s*:\s*(?P<value>.+?)\s*$")

# webui/ops/density_regions.py: one line per image, then a closing tally.
_RE_REGION_DIR = re.compile(r"Writing region overlays to:\s*(?P<path>.+?)\s*$")
_RE_REGION_IMAGE = re.compile(
    r"^\s{2}(?P<name>.+?): total\s+(?P<total>-?[\d.]+)\s+(?P<regions>\d+) regions\s+residual\s+(?P<residual>-?[\d.]+)\s*$"
)
_RE_REGION_TALLY = re.compile(
    r"^(?P<images>\d+) image\(s\), (?P<regions>\d+) regions, (?P<total>-?[\d.]+) chickens total\s*$"
)

_SECTION_TITLES = {"EXHIBITION SUMMARY": "exhibition", "TECHNICAL METRICS": "technical"}

# Run ids look like "train-20260803-155902-3"; the middle two fields are the
# start time, which is more accurate than any file timestamp.
_RE_RUN_ID = re.compile(r"^\w+-(?P<stamp>\d{8}-\d{6})-\d+$")


def _float(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def _started_at_from_id(run_id: str, path: Path) -> float:
    match = _RE_RUN_ID.match(run_id)
    if match:
        try:
            return time.mktime(time.strptime(match["stamp"], "%Y%m%d-%H%M%S"))
        except ValueError:
            pass
    return path.stat().st_mtime


class Run:
    """A single supervised child process plus everything the UI shows about it."""

    _ids = itertools.count(1)

    def __init__(
        self,
        kind: str,
        argv: list[str],
        values: dict,
        run_id: Optional[str] = None,
        env: Optional[dict] = None,
    ):
        self.kind = kind
        self.argv = argv
        self.values = values
        self.env = env
        self.command = " ".join(argv)
        self.id = run_id or f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{next(Run._ids)}"
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.returncode: Optional[int] = None
        self.stopped = False
        self.restored = False

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{self.id}.log"

        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._offset = 0  # absolute index of self._lines[0] after trimming

        self.metrics: dict = {"train": [], "val": [], "best": []}
        # `overlay_suffix` tells the gallery route which PNGs in `overlay_dir`
        # belong to this kind of run; test.py and density_regions.py can write
        # into the same directory.
        self.result: dict = {
            "images": [],
            "exhibition": {},
            "technical": {},
            "overlay_dir": None,
            "overlay_suffix": None,
        }
        # Lets the browser skip re-downloading and re-rendering the complete
        # metrics/result payload when a poll only contains new log lines.
        self.state_version = 0
        self._section: Optional[str] = None
        self._section_rows = 0

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._log_file = None  # open only while the run is live

    # --- lifecycle ---------------------------------------------------------

    @classmethod
    def from_log(cls, path: Path) -> Optional["Run"]:
        """Rebuild a finished run from its log file so history survives a restart.

        Replaying the log through the same parser means restored runs show the
        same charts and evaluation tables as live ones; only the child process
        is gone.
        """
        kind = path.stem.split("-", 1)[0]
        if kind not in ENTRYPOINTS:
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None

        run = cls(kind, [], {}, run_id=path.stem)
        run.restored = True
        if lines and lines[0].startswith("$ "):
            run.command = lines[0][2:]
        run.started_at = _started_at_from_id(path.stem, path)
        run.finished_at = path.stat().st_mtime

        for line in lines:
            run._append(line)
            try:
                run._parse(line)
            except Exception:
                pass

        tail = "\n".join(lines[-3:])
        match = re.search(r"\[webui\] (?P<verb>finished|stopped) with exit code (?P<code>-?\d+)", tail)
        if match:
            run.returncode = int(match["code"])
            run.stopped = match["verb"] == "stopped"
        else:  # the server died while this run was in flight
            run.returncode = -1
            run.stopped = True
        return run

    def start(self) -> None:
        # Everything the UI shows also goes to the log file, so a restored run
        # reads back identically (see from_log).
        self._log_file = open(self.log_path, "w", encoding="utf-8")

        # New process group / session so stopping also reaps dataloader workers.
        kwargs: dict = {}
        if _IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(
            self.argv,
            cwd=ROOT,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **kwargs,
        )
        self._append(f"$ {' '.join(self.argv)}")
        self._append(f"[webui] cwd={ROOT}  pid={self._proc.pid}")
        self._thread = threading.Thread(target=self._pump, name=f"run-{self.id}", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            self._append(line)
            try:
                self._parse(line)
            except Exception:  # a parser bug must never kill the log stream
                pass

        self.returncode = self._proc.wait()
        self.finished_at = time.time()
        verb = "stopped" if self.stopped else "finished"
        self._append(f"[webui] {verb} with exit code {self.returncode}")

        log_file, self._log_file = self._log_file, None
        if log_file is not None:
            log_file.close()

    def stop(self) -> None:
        """Terminate the run and its children (dataloader workers included)."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.stopped = True
        if _IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def status(self) -> str:
        if self.running:
            return "running"
        if self.stopped:
            return "stopped"
        if self.returncode == 0:
            return "done"
        if self.returncode is None:
            return "starting"
        return "failed"

    # --- log buffer --------------------------------------------------------

    def _append(self, line: str) -> None:
        with self._lock:
            if self._log_file is not None:
                self._log_file.write(line + "\n")
                self._log_file.flush()
            self._lines.append(line)
            if len(self._lines) > MAX_LINES:
                drop = len(self._lines) - MAX_LINES
                del self._lines[:drop]
                self._offset += drop

    def lines_since(self, cursor: int, tail: int = 0) -> tuple[list[str], int]:
        """Lines with absolute index >= `cursor`, plus the next cursor value."""
        with self._lock:
            start = max(cursor - self._offset, 0)
            if cursor <= 0 and tail > 0:
                start = max(start, len(self._lines) - tail)
            return self._lines[start:], self._offset + len(self._lines)

    def _touch_state(self) -> None:
        self.state_version += 1

    # --- structured parsing ------------------------------------------------

    def _parse(self, line: str) -> None:
        # Only the entrypoints with a per-image log format get structured
        # results; the rest of the annotation tools just stream summary text.
        if self.kind == "train":
            self._parse_train(line)
        elif self.kind == "test":
            self._parse_test(line)
        elif self.kind == "density_regions":
            self._parse_regions(line)

    def _parse_train(self, line: str) -> None:
        m = _RE_TRAIN.search(line)
        if m:
            self.metrics["train"].append(
                {
                    "epoch": int(m["epoch"]),
                    "lr": _float(m["lr"]),
                    "loss": _float(m["loss"]),
                    "mse": _float(m["mse"]),
                    "mae": _float(m["mae"]),
                }
            )
            self._touch_state()
            return
        m = _RE_VAL.search(line)
        if m:
            self.metrics["val"].append(
                {
                    "epoch": int(m["epoch"]),
                    "tag": m["tag"],
                    "mse": _float(m["mse"]),
                    "mae": _float(m["mae"]),
                }
            )
            self._touch_state()
            return
        m = _RE_BEST.search(line)
        if m:
            self.metrics["best"].append({"name": m["name"], "at": time.time()})
            self._touch_state()

    def _parse_test(self, line: str) -> None:
        m = _RE_OVERLAY_DIR.search(line)
        if m:
            self.result["overlay_dir"] = m["path"]
            self._touch_state()
            return

        stripped = line.strip()
        if stripped in _SECTION_TITLES:
            self._section = _SECTION_TITLES[stripped]
            self._section_rows = 0
            return
        if set(stripped) == {"="} and self._section and self._section_rows:
            self._section = None
            return

        if self._section:
            m = _RE_SUMMARY_KV.match(line)
            if m:
                self.result[self._section][m["key"].strip()] = m["value"].strip()
                self._section_rows += 1
                self._touch_state()
            return

        m = _RE_IMAGE.match(line)
        if m:
            gt, pred = _float(m["gt"]), _float(m["pred"])
            self.result["images"].append(
                {
                    "name": m["name"],
                    "gt": gt,
                    "pred": pred,
                    "err": _float(m["err"].lstrip("+")),
                    "rel": m["rel"],
                }
            )
            self._touch_state()

    def _parse_regions(self, line: str) -> None:
        m = _RE_REGION_DIR.search(line)
        if m:
            self.result["overlay_dir"] = m["path"]
            self.result["overlay_suffix"] = "_regions.png"
            self._touch_state()
            return

        m = _RE_REGION_IMAGE.match(line)
        if m:
            self.result["images"].append(
                {
                    "name": m["name"],
                    "total": _float(m["total"]),
                    "regions": int(m["regions"]),
                    "residual": _float(m["residual"]),
                }
            )
            self._touch_state()
            return

        m = _RE_REGION_TALLY.match(line)
        if m:
            # Same slot the test tool fills, so the UI reads its summary cards
            # from one place regardless of which tool produced them.
            self.result["technical"] = {
                "Images": m["images"],
                "Regions": m["regions"],
                "Chickens (total)": m["total"],
            }
            self._touch_state()

    # --- serialization -----------------------------------------------------

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "elapsed": (self.finished_at or time.time()) - self.started_at,
            "command": self.command,
            "restored": self.restored,
            "log_path": str(self.log_path),
        }

    def detail(self) -> dict:
        data = self.summary()
        data["values"] = self.values
        data["metrics"] = self.metrics
        data["result"] = self.result
        return data


class RunManager:
    """Owns every run started this session and enforces one active run at a time."""

    def __init__(self, restore_limit: int = 50):
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._restore(restore_limit)

    def _restore(self, limit: int) -> None:
        """Load the most recent runs back from logs/webui so history persists."""
        if not LOG_DIR.is_dir():
            return
        logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in logs[:limit]:
            run = Run.from_log(path)
            if run is not None:
                self._runs[run.id] = run

    def start(self, kind: str, argv: list[str], values: dict, env: Optional[dict] = None) -> Run:
        with self._lock:
            active = self.active()
            if active is not None:
                raise RuntimeError(f"run {active.id} is still running; stop it first")
            run = Run(kind, argv, values, env=env)
            while run.id in self._runs:  # never reuse a restored run's log file
                run = Run(kind, argv, values, env=env)
            self._runs[run.id] = run
        run.start()
        return run

    def active(self) -> Optional[Run]:
        return next((r for r in self._runs.values() if r.running), None)

    def get(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    def list(self) -> list[dict]:
        return [r.summary() for r in sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)]

    def clear(self) -> tuple[list[str], list[str]]:
        """Forget every finished run and delete its log file.

        A run still in flight is kept: its log file is open, and dropping the
        entry would leave a live process with no way back to the UI. Only logs
        belonging to runs we know about are removed, so service logs (and any
        stray file in logs/webui/) are left where they are.

        Returns the ids removed and the ids whose log file could not be deleted
        (those stay in history, since history *is* the log directory).
        """
        with self._lock:
            doomed = [run for run in self._runs.values() if not run.running]
            for run in doomed:
                del self._runs[run.id]

        removed, failed = [], []
        for run in doomed:
            try:
                run.log_path.unlink(missing_ok=True)
                removed.append(run.id)
            except OSError:
                failed.append(run.id)
                with self._lock:
                    self._runs[run.id] = run
        return removed, failed

    def stop_all(self) -> None:
        for run in self._runs.values():
            if run.running:
                run.stop()
