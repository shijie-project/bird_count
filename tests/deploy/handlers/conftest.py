import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from birdcount.deploy.alarm import load_config
from birdcount.deploy.handlers.base import Batch
from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.results import empty_batch


CAMERA_ID = "axis1/B8A44FD52B7F"
MAC = "B8A44FD52B7F"
THRESHOLD = 140

# Compressed rules so a whole event runs in a few simulated seconds.
ALARM_CONFIG = {
    "output_dir": "runs",
    "sample_fps": 5.0,
    "rules": {
        "alert_trigger_seconds": 2.0,
        "escalation_seconds": 3.0,
        "recovery_seconds": 2.0,
        "evidence_window_seconds": 5.0,
        "max_level": 3,
    },
    "motion_filter": {"enabled": False},
    "worker": {
        "enabled": True,
        "dry_run": True,
        "base_url": "https://example.test",
        "project_slug": "test-farm",
        "mock_recipients": ["+61400000000"],
    },
    "cameras": [{"camera_id": CAMERA_ID, "name": "test camera", "threshold": THRESHOLD}],
}

# Over threshold long enough for L1 (t=3), L2 (t=6), L3 (t=9), then clear.
ESCALATION = [(0, 90), (1, 200), (2, 200), (3, 200), (6, 200), (9, 200), (10, 200), (11, 50), (12, 50), (13, 50)]


def write_config(tmp: Path, **overrides):
    path = tmp / "alarm.json"
    path.write_text(json.dumps({**ALARM_CONFIG, **overrides}), encoding="utf-8")
    return load_config(path)


@pytest.fixture
def alarm_config(tmp_path):
    return write_config(tmp_path)


@pytest.fixture
def out():
    """A short output root: evidence paths nest the event id twice, and pytest's
    own tmp_path would push them past Windows' 260-character limit."""
    path = Path(tempfile.mkdtemp(prefix="bc"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def ring():
    """Stands in for FrameRing: handlers only read `.frames[sid, slot]`."""
    return SimpleNamespace(frames=np.full((2, 2, 72, 128, 3), 200, np.uint8))


def records(sid: int, ts: float, count: float, cx=float("nan"), cy=float("nan")) -> np.ndarray:
    r = empty_batch(1)
    r["sid"], r["slot"], r["captured"], r["count"], r["cx"], r["cy"] = sid, 0, ts, count, cx, cy
    return r


def run(engine: AlarmEngine, handlers, ring, series, t0, sid=0, hijacked=lambda t: set()):
    """Feed (offset, count) samples through the engine and handlers as the consumer does."""
    batches = []
    for offset, count in series:
        recs = records(sid, t0 + offset, count)
        decisions, manual = engine.evaluate(recs, hijacked(offset))
        batch = Batch(recs, decisions, manual, ring)
        for h in handlers:
            h.handle(batch)
        batches.append(batch)
    return batches
