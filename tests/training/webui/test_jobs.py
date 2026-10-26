"""Jobs: real child processes, stopped as Ctrl+C would stop them, killed when they will not stop."""

import os
import sys
import time
from pathlib import Path

import pytest

from birdcount.training import child
from birdcount.training.webui.jobs import Busy, Jobs


TOOL = """
import sys, time
import numpy  # loads DLLs: a thread blocked on a stdin pipe would deadlock this
print("ready", flush=True)
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    print("saving state", flush=True)
    sys.exit(130)
"""

STUBBORN = """
import time
print("ready", flush=True)
while True:
    try:
        time.sleep(0.05)
    except KeyboardInterrupt:
        print("ignoring the interrupt", flush=True)
"""


@pytest.fixture
def env(tmp_path) -> dict:
    (tmp_path / "mods").mkdir()
    (tmp_path / "mods" / "demo_tool.py").write_text(TOOL, encoding="utf-8")
    (tmp_path / "mods" / "demo_stubborn.py").write_text(STUBBORN, encoding="utf-8")
    return {**os.environ, "PYTHONPATH": str(tmp_path / "mods")}


def wait_for(path: Path, text: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and text in path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(0.05)
    raise AssertionError(
        f"{text!r} never appeared in {path}:\n{path.read_text(errors='replace') if path.is_file() else ''}"
    )


def test_a_stopped_job_cleans_up_like_on_ctrl_c(tmp_path, env) -> None:
    jobs = Jobs(tmp_path / "jobs")
    job = jobs.start("demo", "train", child.command("demo_tool"), run_dir=None, gpu=True, env=env)
    wait_for(job.log_path, "ready")
    assert job.status == "running" and [j.id for j in jobs.running()] == [job.id]
    with pytest.raises(Busy):
        jobs.start("demo", "train", child.command("demo_tool"), run_dir=None, gpu=True, env=env)
    job.stop(grace=30)
    assert job.wait(30)
    time.sleep(0.2)  # the waiter thread writes the record
    log = job.log_path.read_text(encoding="utf-8")
    assert "saving state" in log and "[panel] stopped, exit code 130" in log
    assert job.status == "stopped" and job.record["returncode"] == 130
    assert not job.stop_file.exists()

    reloaded = Jobs(tmp_path / "jobs")  # history survives a restart of the panel
    assert reloaded.get(job.id).status == "stopped"
    assert reloaded.clear() == 1 and not job.log_path.exists()


def test_a_job_that_ignores_the_stop_is_killed_after_the_grace(tmp_path, env) -> None:
    jobs = Jobs(tmp_path / "jobs")
    job = jobs.start("demo", "train", child.command("demo_stubborn"), run_dir=None, gpu=False, env=env)
    wait_for(job.log_path, "ready")
    t0 = time.monotonic()
    job.stop(grace=1.0)
    assert job.wait(30)
    assert time.monotonic() - t0 < 15
    wait_for(job.log_path, "killing it")
    assert "ignoring the interrupt" in job.log_path.read_text(encoding="utf-8")


def test_a_job_whose_panel_vanished_is_reported_lost(tmp_path) -> None:
    folder = tmp_path / "jobs"
    folder.mkdir()
    (folder / "x-1.json").write_text(
        '{"id": "x-1", "tool": "x", "page": "train", "argv": [], "command": "",'
        ' "started": 1, "finished": null, "returncode": null, "stopped": false}'
    )
    job = Jobs(folder).get("x-1")
    assert job.status == "lost" and job.to_json()["returncode"] == -1


def test_the_tool_runs_as_python_m_would(tmp_path, env) -> None:
    (tmp_path / "mods" / "demo_args.py").write_text(
        "import sys\nprint('argv', sys.argv[1:], __name__)\nsys.exit(3)\n", encoding="utf-8"
    )
    jobs = Jobs(tmp_path / "jobs")
    job = jobs.start("demo", "data", child.command("demo_args", "--x", "a b"), run_dir=None, gpu=False, env=env)
    assert job.wait(60)
    time.sleep(0.2)
    assert "argv ['--x', 'a b'] __main__" in job.log_path.read_text(encoding="utf-8")
    assert job.status == "failed" and job.record["returncode"] == 3
    assert child.command("m")[:4] == [sys.executable, "-u", "-m", "birdcount.training.child"]
