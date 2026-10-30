from datetime import datetime
from pathlib import Path

from birdcount.deploy.replay.recordings import clip_start, days_of, find_cameras


MAC = "B8A44FD51C3C"


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def archive(root: Path) -> Path:
    """The AXIS export's layout: two recording groups on one day, one on the next, and clutter."""
    cam = root / "axis1" / f"axis-{MAC}"
    group = cam / "20260311" / "09" / f"20260311_091325_66D9_{MAC}"
    touch(group / "20260311_10" / "20260311_100330_9A81.mkv")
    touch(group / "20260311_09" / "20260311_091325_972C.mkv")
    touch(group / "20260311_09" / "20260311_091825_18E4.mkv")
    touch(group / "20260311_09" / "20260311_091825_18E4_overlay.mp4")  # a render, not a clip
    touch(cam / "20260311" / "12" / f"20260311_121414_E552_{MAC}" / "20260311_12" / "20260311_121414_ABF7.mkv")
    touch(cam / "20260312" / "05" / f"20260312_055956_5010_{MAC}" / "20260312_05" / "20260312_055956_1111.mkv")
    touch(cam / "recording_groups" / "recording_groups.conf")
    touch(cam / "index.db")
    touch(root / "axis2" / "axis-E82725003277" / "20260311" / "05" / "g" / "h" / "20260311_050000_AAAA.mkv")
    (root / "code" / "data").mkdir(parents=True)
    return cam


def test_cameras_are_found_from_the_archive_a_recorder_or_the_camera_itself(tmp_path) -> None:
    cam = archive(tmp_path)
    assert list(find_cameras(tmp_path)) == [MAC, "E82725003277"]
    assert list(find_cameras(tmp_path / "axis1")) == [MAC]
    found = find_cameras(cam)
    assert list(found) == [MAC]
    assert found[MAC].dirs == (cam,)
    assert found[MAC].group == "axis1"


def test_a_day_is_its_clips_in_start_order_across_groups(tmp_path) -> None:
    archive(tmp_path)
    days = days_of(find_cameras(tmp_path)[MAC])
    assert list(days) == ["2026-03-11", "2026-03-12"]
    assert [c.path.name for c in days["2026-03-11"]] == [
        "20260311_091325_972C.mkv",
        "20260311_091825_18E4.mkv",
        "20260311_100330_9A81.mkv",
        "20260311_121414_ABF7.mkv",
    ]
    assert days["2026-03-11"][0].start == datetime(2026, 3, 11, 9, 13, 25)


def test_a_clip_exported_twice_counts_once(tmp_path) -> None:
    cam = archive(tmp_path)
    touch(tmp_path / "axis1-copy" / f"axis-{MAC}" / "20260312" / "05" / "g" / "h" / "20260312_055956_1111.mkv")
    found = find_cameras(tmp_path)
    assert len(found[MAC].dirs) == 2
    assert [c.path.name for c in days_of(found[MAC])["2026-03-12"]] == ["20260312_055956_1111.mkv"]
    assert cam in found[MAC].dirs


def test_only_recorder_clips_have_a_start() -> None:
    assert clip_start(Path("20260311_091325_972C.mkv")) == datetime(2026, 3, 11, 9, 13, 25)
    assert clip_start(Path("20260311_091325_972C_overlay.mp4")) is None
    assert clip_start(Path(f"{MAC}_20260311_091325_972C.mkv")) is None
    assert clip_start(Path("20261399_091325_972C.mkv")) is None
