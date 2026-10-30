import threading

import pytest

from birdcount.deploy.replay.cache import ClipCache, InPlace, Stopped


def sources(tmp_path, n: int, tag: str = "c"):
    out = []
    for i in range(n):
        path = tmp_path / "rec" / f"{tag}{i}.mkv"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"{tag}{i}".encode() * 100)
        out.append(path)
    return out


def test_clips_arrive_as_local_copies_and_go_once_counted(tmp_path) -> None:
    clips = sources(tmp_path, 5)
    cache = ClipCache(tmp_path / "cache", ahead=2)
    plan = cache.open(clips)
    try:
        for i, src in enumerate(clips):
            local = plan.fetch(i)
            assert len(list(local.parent.iterdir())) <= 2  # never more than `ahead` clips per camera-day
            assert local.parent == tmp_path / "cache"
            assert local.read_bytes() == src.read_bytes()
            plan.done(i)
            assert not local.exists()
    finally:
        plan.close()
        cache.close()
    assert not (tmp_path / "cache").exists()
    assert all(src.exists() for src in clips)  # the recordings are only read


def test_a_clip_that_cannot_be_copied_is_an_error_for_that_clip_only(tmp_path) -> None:
    clips = sources(tmp_path, 3)
    clips[1] = tmp_path / "rec" / "missing.mkv"
    cache = ClipCache(tmp_path / "cache", ahead=3)
    plan = cache.open(clips)
    try:
        plan.fetch(0)
        plan.done(0)
        with pytest.raises(OSError, match=r"missing\.mkv"):
            plan.fetch(1)
        plan.done(1)
        assert plan.fetch(2).exists()
    finally:
        plan.close()
        cache.close()


def test_camera_days_are_served_side_by_side(tmp_path) -> None:
    cache = ClipCache(tmp_path / "cache", ahead=1)
    a, b = cache.open(sources(tmp_path, 3, "a")), cache.open(sources(tmp_path, 3, "b"))
    try:
        for i in range(3):
            for plan in (a, b):
                assert plan.fetch(i).exists()
                plan.done(i)
    finally:
        a.close()
        b.close()
        cache.close()


def test_waiting_for_a_clip_gives_up_when_the_replay_stops(tmp_path) -> None:
    cache = ClipCache(tmp_path / "cache", ahead=1)
    plan = cache.open(sources(tmp_path, 2))
    stop = threading.Event()
    stop.set()
    plan.fetch(0)  # copied by now: a stop does not take back what is ready
    with pytest.raises(Stopped):
        plan.fetch(1, stop)  # not copied yet: with ahead=1 only clip 0 was wanted
    plan.close()
    cache.close()


def test_clips_already_on_a_fast_disk_are_read_in_place(tmp_path) -> None:
    clips = sources(tmp_path, 2)
    plan = InPlace(clips)
    assert plan.fetch(1) == clips[1]
    plan.done(1)
    assert clips[1].exists()
