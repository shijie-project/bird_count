"""The whole runtime on video files, CPU inference: start, keep a dying worker alive, stop cleanly."""

import json
import threading
import time
from dataclasses import replace

import pytest
import torch

from birdcount.core.model import DensityNet
from birdcount.core.paths import PROJECT_ROOT
from birdcount.deploy import app as app_module
from birdcount.deploy.app import AlreadyRunning, App
from birdcount.deploy.settings import DeploySettings
from birdcount.deploy.shm import FrameRing
from birdcount.deploy.streams import resolve_streams
from birdcount.deploy.topology import parse_topology

from .conftest import H, W, make_layout, write_video


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "RING_PREFIX", make_layout().prefix)  # never the real runtime's ring
    ckpt = tmp_path / "m.pth"
    torch.save(DensityNet().state_dict(), ckpt)
    clips = tmp_path / "clips"
    clips.mkdir()
    for i in range(2):
        write_video(clips / f"cam{i}.avi", 40)
    off = {f"enable_{k}": False for k in ("monitor", "speaker", "smart_plug", "sms_alarm", "count_logger")}
    settings = DeploySettings(
        **off,
        model_path=ckpt,
        image_width=W,
        image_height=H,
        fps=5,
        source_type="video",
        video_paths=str(clips),
        num_streams=None,
        debug=False,
        cuda_device="cpu",
        num_workers_per_gpu=1,
        hw_decode=False,
        mask_dir=None,
        alarm_config_path=PROJECT_ROOT / "configs" / "alarm.json",
        enable_video_recorder=True,
        video_record_dir=tmp_path / "rec",
        audit_log_path=tmp_path / "audit.jsonl",
        log_dir=tmp_path / "logs",
    )
    streams = resolve_streams(settings, parse_topology({"zones": [{"name": "Z"}]}))
    return App(settings, streams, panels=False), tmp_path


def _wait(cond, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


@pytest.mark.slow
def test_runs_restarts_a_dead_worker_and_stops_cleanly(runtime) -> None:
    app, tmp = runtime
    seen = {}

    def operator():
        try:
            assert _wait(lambda: app.consumer is not None and app.consumer.batches >= 5, 90)
            victim = app.inferencers[0].proc
            victim.kill()
            assert _wait(lambda: app.inferencers[0].proc is not victim, 10)
            restarted_at = app.consumer.batches
            assert _wait(lambda: app.consumer.batches >= restarted_at + 5, 90)
            seen["ok"] = True
        finally:
            app.request_stop("test")

    threading.Thread(target=operator, daemon=True).start()
    assert app.run() == 0
    assert seen.get("ok"), "results did not resume after the worker was restarted"
    assert [c.restarts for c in app.children] == [1, 0]
    assert not any(c.proc.is_alive() for c in app.children)
    assert app.stopped.is_set()

    events = [json.loads(line)["event"] for line in (tmp / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0] == "system.start" and events[-1] == "system.stop"
    assert "system.process_restart" in events
    assert events.count("recorder.segment_open") == events.count("recorder.segment_close") >= 2
    FrameRing.allocate(app.ring.layout, fps=5).close()  # the ring was freed


def test_a_second_runtime_refuses_to_start(runtime) -> None:
    app, _ = runtime
    other = FrameRing.allocate(replace(make_layout(), prefix=app_module.RING_PREFIX), fps=5)
    try:
        with pytest.raises(AlreadyRunning):
            app.run()
    finally:
        other.close()


def test_check_lists_the_streams_and_flags_bad_configuration(tmp_path, monkeypatch, capsys) -> None:
    from birdcount.deploy.__main__ import main

    (tmp_path / "B8A44FD51C3C_day1.mkv").write_bytes(b"")
    (tmp_path / "m.pth").write_bytes(b"")
    for key, value in {
        "SOURCE_TYPE": "video",
        "VIDEO_PATHS": str(tmp_path),
        "NUM_STREAMS": "",
        "MODEL_PATH": str(tmp_path / "m.pth"),
        "MASK_DIR": "",
        "ALARM_CONFIG_PATH": str(PROJECT_ROOT / "configs" / "alarm.json"),
    }.items():
        monkeypatch.setenv(key, value)
    assert main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "B8A44FD51C3C" in out and "1 stream(s); 1 without both a threshold and a mask" in out

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pth"))
    assert main(["--check"]) == 1
