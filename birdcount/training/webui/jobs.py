"""The processes the panel starts: one tool run each, stopped the way Ctrl+C would.

A job runs `python -m birdcount.training.child <tool module> ...`, writing its
console output straight into a log file, which the page reads back by byte
offset — the panel parses none of it: results come from the run folder the
tool writes (`results`). Each job also keeps a small JSON record next to its
log, so the history survives a restart of the panel.

Stopping asks the tool to stop (`child.stop`: its stop file appears), and
kills the whole process tree only if it has not finished within the grace period.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from birdcount.core.paths import PROJECT_ROOT
from birdcount.training import child


GRACE_SECONDS = 30.0  # a training run saves its state; a video analysis writes its timeline
HISTORY = 200  # job records loaded at startup

_IS_WINDOWS = os.name == "nt"


def spawn(argv: list[str], log_path: Path, env: dict | None = None) -> subprocess.Popen:
    """Start `argv` in its own process group, its output appended to `log_path`."""
    kwargs: dict = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if _IS_WINDOWS else {"start_new_session": True}
    )
    with log_path.open("ab") as log:
        return subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            env={**(env or os.environ), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **kwargs,
        )


def kill_tree(pid: int) -> bool:
    """Kill a process and everything it started. True when the kill went through."""
    try:
        if _IS_WINDOWS:
            done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            return done.returncode == 0
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        return True
    except OSError:
        return False


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


class Job:
    _ids = itertools.count(1)

    def __init__(self, folder: Path, record: dict):
        self.folder = folder
        self.record = record
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def id(self) -> str:
        return self.record["id"]

    @property
    def log_path(self) -> Path:
        return self.folder / f"{self.id}.log"

    @property
    def stop_file(self) -> Path:
        return self.folder / f"{self.id}.stop"

    @property
    def run_dir(self) -> Path | None:
        return Path(self.record["run_dir"]) if self.record.get("run_dir") else None

    @classmethod
    def new(cls, folder: Path, tool: str, page: str, argv: list[str], *, run_dir: Path | None, gpu: bool) -> Job:
        job_id = f"{tool}-{time.strftime('%Y%m%d-%H%M%S')}-{next(cls._ids)}"
        record = {
            "id": job_id,
            "tool": tool,
            "page": page,
            "argv": argv,
            "command": subprocess.list2cmdline(argv) if _IS_WINDOWS else " ".join(argv),
            "run_dir": str(run_dir) if run_dir else None,
            "gpu": gpu,
            "started": time.time(),
            "finished": None,
            "returncode": None,
            "stopped": False,
        }
        return cls(folder, record)

    @classmethod
    def load(cls, path: Path) -> Job | None:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if record.get("finished") is None:  # the panel went away while it ran
            record.update(finished=path.stat().st_mtime, returncode=-1, lost=True)
        return cls(path.parent, record)

    def _save(self) -> None:
        tmp = self.folder / f"{self.id}.json.tmp"
        tmp.write_text(json.dumps(self.record, indent=1), encoding="utf-8")
        os.replace(tmp, self.folder / f"{self.id}.json")

    # --- lifecycle --------------------------------------------------------------------------

    def start(self, env: dict | None = None) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(f"$ {self.record['command']}\n", encoding="utf-8")
        self._proc = spawn(self.record["argv"], self.log_path, child.env(self.stop_file, env))
        self.record["pid"] = self._proc.pid
        self._save()
        threading.Thread(target=self._wait, name=f"job-{self.id}", daemon=True).start()

    def _wait(self) -> None:
        code = self._proc.wait()
        with self._lock:
            self.record.update(finished=time.time(), returncode=code)
            verb = "stopped" if self.record["stopped"] else "finished"
            append_line(self.log_path, f"[panel] {verb}, exit code {code}")
            self._save()
        self.stop_file.unlink(missing_ok=True)

    def stop(self, grace: float = GRACE_SECONDS) -> None:
        """Ask the tool to stop; kill it if it is still running after `grace` s."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        with self._lock:
            self.record["stopped"] = True
        child.stop(self.stop_file)
        threading.Thread(target=self._kill_after, args=(grace,), daemon=True).start()

    def _kill_after(self, grace: float) -> None:
        try:
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            append_line(self.log_path, f"[panel] still running after {grace:.0f}s; killing it")
            kill_tree(self._proc.pid)

    def wait(self, timeout: float | None = None) -> bool:
        """True once the process has exited."""
        if self._proc is None:
            return True
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def status(self) -> str:
        r = self.record
        if self.running:
            return "running"
        if r.get("lost"):
            return "lost"
        if r["stopped"]:
            return "stopped"
        if r["returncode"] is None:
            return "starting"
        return "done" if r["returncode"] == 0 else "failed"

    def to_json(self) -> dict:
        r = self.record
        return {
            "id": r["id"],
            "tool": r["tool"],
            "page": r["page"],
            "command": r["command"],
            "run_dir": r.get("run_dir"),
            "gpu": r.get("gpu", False),
            "started": r["started"],
            "finished": r.get("finished"),
            "returncode": r.get("returncode"),
            "status": self.status,
        }


class Busy(RuntimeError):
    pass


class Jobs:
    """Every job of this panel, past ones from their records. One GPU job at a time."""

    def __init__(self, folder: Path):
        self.folder = folder
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        records = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
        for path in records[-HISTORY:]:
            job = Job.load(path)
            if job is not None:
                self._jobs[job.id] = job

    def start(
        self, tool: str, page: str, argv: list[str], *, run_dir: Path | None, gpu: bool, env: dict | None = None
    ) -> Job:
        with self._lock:
            if gpu and (busy := next((j for j in self._jobs.values() if j.running and j.record.get("gpu")), None)):
                raise Busy(f"{busy.id} is using the GPU; stop it first")
            job = Job.new(self.folder, tool, page, argv, run_dir=run_dir, gpu=gpu)
            self._jobs[job.id] = job
        job.start(env)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.record["started"], reverse=True)

    def running(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.running]

    def by_run_dir(self) -> dict[Path, Job]:
        """The newest job writing each run folder (and the folders inside it)."""
        out: dict[Path, Job] = {}
        for job in sorted(self._jobs.values(), key=lambda j: j.record["started"]):
            if job.run_dir is not None:
                out[job.run_dir] = job
        return out

    def stop_all(self, grace: float = GRACE_SECONDS) -> None:
        """Stop every running job and wait for them (at most `grace` s, then kill)."""
        live = self.running()
        for job in live:
            job.stop(grace)
        deadline = time.monotonic() + grace + 5
        for job in live:
            job.wait(max(0.0, deadline - time.monotonic()))

    def clear(self) -> int:
        """Forget finished jobs and delete their logs; running ones stay."""
        removed = 0
        with self._lock:
            for job in [j for j in self._jobs.values() if not j.running]:
                for path in (job.log_path, job.folder / f"{job.id}.json"):
                    path.unlink(missing_ok=True)
                del self._jobs[job.id]
                removed += 1
        return removed
