"""Run folders read back: where the tools put runs, and what the page is told about them."""

import os
import time
from datetime import datetime
from pathlib import Path

from birdcount.training import runs
from birdcount.training.webui import results
from birdcount.training.webui.dataset import Dataset

from .conftest import make_train, make_video


NOW = datetime(2026, 10, 26, 13, 45, 7)


def test_run_folder_names(tmp_path) -> None:
    assert runs.train_dir(tmp_path, "x", NOW) == tmp_path / "train" / "20261026-134507-x"
    assert runs.train_dir(tmp_path, now=NOW).name == "20261026-134507"
    assert runs.sweep_dir(tmp_path, NOW) == tmp_path / "train" / "sweep-20261026-134507"
    assert (
        runs.eval_dir(tmp_path, "../ckpts/best.pth", "val", NOW) == tmp_path / "eval" / "best" / "val-20261026-134507"
    )
    assert runs.video_dir(tmp_path, Path("d/clip.mkv"), "S") == tmp_path / "video" / "clip" / "S"
    assert runs.started(tmp_path / "val-20261026-134507") == NOW.timestamp()


def test_runs_are_found_by_kind(output_root) -> None:
    names = {kind: [p.name for p in runs.find(output_root, kind)] for kind in runs.KINDS}
    assert names == {
        "train": ["20261002-100000-b", "20261001-100000-a"],  # newest first; sweep runs live in their sweep
        "sweep": ["sweep-20261003-100000"],
        "eval": ["val-20261004-100000"],
        "video": ["20261005-100000"],
    }
    for kind in runs.KINDS:
        for path in runs.find(output_root, kind):
            assert runs.kind_of(path) == kind
    batch = output_root / "video" / "20261006-batch"
    make_video(batch / "one")
    make_video(batch / "two")
    assert [p.name for p in runs.under(batch)] == ["one", "two"]
    assert runs.under(batch / "one") == [batch / "one"]
    assert runs.under(output_root / "nothing") == []


def test_line_files_are_read_from_an_offset_whole_lines_only(tmp_path) -> None:
    path = tmp_path / "m.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a"')
    records, offset = results.read_jsonl(path, 0)
    assert records == [{"a": 1}, {"a": 2}] and offset == 18
    with path.open("ab") as f:
        f.write(b": 3}\n")
    assert results.read_jsonl(path, offset) == ([{"a": 3}], path.stat().st_size)
    assert results.read_jsonl(tmp_path / "missing", 5) == ([], 5)


def test_logs_start_at_their_tail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(results, "LOG_TAIL", 10)
    path = tmp_path / "x.log"
    path.write_text("0123456789abcdefghij", encoding="utf-8")
    first = results.read_log(path, -1)
    assert first == {"text": "abcdefghij", "offset": 20}
    with path.open("a") as f:
        f.write("XY")
    assert results.read_log(path, first["offset"]) == {"text": "XY", "offset": 22}
    assert results.read_log(tmp_path / "nope.log", -1)["missing"]


def test_train_runs(output_root) -> None:
    done = results.detail(output_root / "train" / "20261001-100000-a", "train", live=False)
    assert done["status"] == "done" and done["epochs"] == [4, 10] and done["best"]["counting_accuracy"] == 0.9
    assert [r["epoch"] for r in done["records"]] == [0, 1, 2, 3] and done["checkpoints"] == [
        "best_ep0003_mae3.70_rmse5.00.pth"
    ]

    going = output_root / "train" / "20261002-100000-b"
    assert results.overview(going, "train", live=True)["status"] == "running"
    info = results.overview(going, "train", live=False)
    assert info["status"] == "running"  # written to moments ago: maybe a run started from a shell
    assert info["best"] == {"epoch": 3, "mae": 3.7, "rmse": 5.0, "counting_accuracy": None, "primary_accuracy": None}
    old = time.time() - 3600
    for p in going.iterdir():
        os.utime(p, (old, old))
    assert results.overview(going, "train", live=False)["status"] == "incomplete"
    d = results.detail(going, "train", live=False)
    assert len(d["records"]) == 4  # the half-written line waits
    assert results.detail(going, "train", live=False, offset=d["offset"])["records"] == []


def test_sweep_eval_and_video_runs(output_root) -> None:
    sweep = results.detail(output_root / "train" / "sweep-20261003-100000", "sweep", live=False)
    assert sweep["status"] == "done" and [r["run"] for r in sweep["leaderboard"]] == ["lr=1"]
    make_train(output_root / "train" / "sweep-20261003-100000" / "lr=2", finished=False)
    (output_root / "train" / "sweep-20261003-100000" / "lr=2" / "summary.json").write_text('{"status": "interrupted"}')
    assert (
        results.overview(output_root / "train" / "sweep-20261003-100000", "sweep", live=False)["status"] == "stopped"
    )

    ev = results.detail(output_root / "eval" / "best" / "val-20261004-100000", "eval", live=False)
    assert ev["status"] == "done" and ev["split"] == "val" and ev["counting_accuracy"] == 0.9
    assert [(i["name"], i["err"]) for i in ev["images"]] == [("a", 2.0), ("b", -5.0)]
    assert ev["regions"]["blobs"]["total"] == 2 and ev["regions"]["blobs"]["worst"][0]["error"] == 2.0
    assert ev["overlays"]["kinds"] == ["blobs", "counts"]

    video = results.detail(output_root / "video" / "clip" / "20261005-100000", "video", live=False)
    assert video["status"] == "done" and video["max_count"] == 9.0 and video["samples"][1] == [1, 1.0, 6.0, 1.0]
    live = make_video(output_root / "video" / "clip" / "20261007-100000", 3, status="running")
    assert results.overview(live, "video", live=True)["status"] == "running"


def test_the_dataset_as_the_browser_sees_it(data_root) -> None:
    ds = Dataset(data_root)
    assert ds.splits() == [{"name": "val", "images": 2}]
    images = ds.images("val")["images"]
    assert [(i["name"], i["gt"], i["annotated"]) for i in images] == [("a", 2, True), ("b", 0, False)]
    assert ds.points("val", "a") == [[1.0, 2.0], [3.0, 4.0]] and ds.points("val", "zzz") == []
    for bad in ("../x", "..\\x", ".hidden", ""):
        try:
            ds.points(bad, "a")
        except ValueError:
            continue
        raise AssertionError(f"split {bad!r} was accepted")


def test_ended_runs_are_read_again_only_when_their_files_change(output_root, monkeypatch) -> None:
    run = output_root / "train" / "20261001-100000-a"
    calls, real = [], results.overview
    monkeypatch.setattr(results, "overview", lambda *a: calls.append(a) or real(*a))
    assert results.overview_cached(run, "train", False)["status"] == "done"
    assert results.overview_cached(run, "train", False)["status"] == "done" and len(calls) == 1
    (run / "summary.json").write_text('{"status": "interrupted", "best": null}', encoding="utf-8")
    assert results.overview_cached(run, "train", False)["status"] == "stopped" and len(calls) == 2
    results.overview_cached(run, "train", True)  # a live run is always read
    assert len(calls) == 3
