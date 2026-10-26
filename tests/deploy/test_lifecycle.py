"""A child process stops by itself when the process that started it dies."""

import subprocess
import sys
import time

import psutil
import pytest


# A parent that starts a frame grabber (no cameras: it just idles), reports
# the grabber's pid, then waits to be killed.
PARENT = """
import multiprocessing as mp, sys, time, uuid
from birdcount.deploy.grabber import GrabberProcess
from birdcount.deploy.shm import FrameRing, RingLayout
if __name__ == "__main__":
    ring = FrameRing.allocate(RingLayout(f"bc_test_{uuid.uuid4().hex[:8]}", 1, 3, (64, 48)), fps=5)
    child = GrabberProcess(ring.layout, [], log_level="WARNING")
    child.start()
    print(child.pid, flush=True)
    time.sleep(120)
"""


@pytest.mark.slow
def test_an_orphaned_child_exits_within_seconds() -> None:
    parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
    try:
        child = psutil.Process(int(parent.stdout.readline()))
        time.sleep(1.0)  # let it reach its run loop
        assert child.is_running()
        # sys.executable may be a venv launcher; kill the real interpreter too.
        for p in psutil.Process(parent.pid).children(recursive=True):
            if p.pid != child.pid:
                p.kill()
        parent.kill()
        child.wait(timeout=5)  # psutil.TimeoutExpired if the orphan lives on
    finally:
        parent.kill()
