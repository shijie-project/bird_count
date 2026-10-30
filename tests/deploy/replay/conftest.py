"""A replayed camera-day: scripted counts over real (tiny) clips, through the real alarm, count log and evidence."""

import json
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from birdcount.core.inference import Prediction
from birdcount.deploy.alarm import load_config
from birdcount.deploy.replay.day import DayJob, Outputs, replay_day
from birdcount.deploy.replay.layout import RunLayout
from birdcount.deploy.replay.recordings import Clip


MAC = "B8A44FD52B7F"
DAY = "2026-03-12"
THRESHOLD = 140
SIZE = (64, 48)  # (width, height)
CLIP_FPS = 10.0
FPS = 5  # the runtime's: every 2nd frame of a 10 fps clip
FRAMES = 60  # 6 s a clip, 30 samples

# Compressed rules so a whole event plays out in a few seconds of video.
ALARM_CONFIG = {
    "output_dir": "runs",
    "sample_fps": 5.0,
    "rules": {
        "alert_trigger_seconds": 2.0,
        "escalation_seconds": 3.0,
        "recovery_seconds": 2.0,
        "evidence_window_seconds": 5.0,
        "max_level": 3,
        "recovery_ratio": 1.0,
        "smoothing_seconds": 0.0,
    },
    "motion_filter": {"enabled": False},
    "worker": {"enabled": True, "dry_run": True, "base_url": "https://example.test", "mock_recipients": []},
    "cameras": [{"camera_id": f"axis1/{MAC}", "name": "test camera", "threshold": THRESHOLD}],
}


class ScriptedPredictor:
    """DensityPredictor's surface, counting what `count_of(sample number)` says instead of a model."""

    def __init__(self, count_of, max_batch: int = 4):
        self.count_of = count_of
        self.max_batch = max_batch
        self.frame_size = SIZE
        self.masks = None
        self.staging = np.zeros((max_batch, SIZE[1], SIZE[0], 3), np.uint8)
        self.seen = 0

    def run(self, n: int, mask_ids=None, *, want_density: bool = False) -> Prediction:
        counts = np.array([self.count_of(self.seen + i) for i in range(n)], np.float32)
        self.seen += n
        return Prediction(counts=counts, centroids=np.full((n, 2), 0.5, np.float32), density=None)


@pytest.fixture
def out():
    """A short root: evidence paths nest the event id twice, past Windows' limit under pytest's tmp_path."""
    path = Path(tempfile.mkdtemp(prefix="bc"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def config_path(out):
    path = out / "alarm.json"
    path.write_text(json.dumps(ALARM_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def config(config_path):
    return load_config(config_path)


@pytest.fixture
def clips(out, make_video):
    """Three back-to-back 6 s clips from 10:00:00."""
    made = []
    for k in range(3):
        path = out / "rec" / f"20260312_1000{6 * k:02d}_00A{k}.mkv"
        path.parent.mkdir(parents=True, exist_ok=True)
        make_video(path, FRAMES, SIZE, CLIP_FPS)
        made.append(Clip(path, datetime(2026, 3, 12, 10, 0, 6 * k)))
    return made


def replay(layout: RunLayout, clips, predictor, config, *, stop=None):
    layout.clear(MAC, DAY)
    outputs = Outputs(layout.count_log, layout.evidence_of(MAC, DAY, THRESHOLD), layout.samples_of(MAC, DAY))
    record = replay_day(
        DayJob(MAC, DAY, tuple(clips)),
        predictor,
        config,
        outputs,
        fps=FPS,
        meta={"source": "replay"},
        model_path=None,
        stop=stop or threading.Event(),
    )
    if record is not None:
        layout.finish(MAC, DAY, record)
    return record


def piled_up_for_two_clips(i: int) -> float:
    return 200.0 if i < 60 else 10.0
