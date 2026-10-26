from datetime import datetime

import numpy as np
import pytest

from birdcount.training.video.timeline import (
    Sample,
    clock_of,
    parse_start_time,
    pick_peaks,
    rolling_mean,
    start_from_name,
    summarize,
    window_peak,
    write_csv,
)


def sample(i: int, t: float, count: float = 0.0, peak: float = 0.0) -> Sample:
    return Sample(i, int(t * 10), t, count, peak, (0, 0, 8, 8))


def test_window_peak_is_the_birds_inside_the_busiest_patch() -> None:
    density = np.zeros((10, 12), np.float16)
    density[4:6, 7:9] = 0.5  # 2 birds in four cells
    density[0, 0] = 0.25  # a lone bird elsewhere
    value, box = window_peak(density, window=4, stride=8)
    assert value == pytest.approx(2.0)
    x0, y0, x1, y1 = box
    assert (x1 - x0, y1 - y0) == (32, 32)
    assert x0 <= 7 * 8 and x1 >= 9 * 8 and y0 <= 4 * 8 and y1 >= 6 * 8  # the box holds the mass


def test_a_window_larger_than_the_map_reports_the_total() -> None:
    density = np.full((3, 5), 0.25, np.float32)
    value, box = window_peak(density, window=16, stride=8)
    assert value == pytest.approx(3.75)
    assert box == (0, 0, 40, 24)


def test_peaks_are_the_highest_moments_kept_apart_in_time_order() -> None:
    samples = [sample(i, float(i), peak=v) for i, v in enumerate([1, 9, 8, 2, 7, 1, 6])]
    chosen = pick_peaks(samples, 3, gap=2.0, key="peak")
    assert [s.t for s in chosen] == [1.0, 4.0, 6.0]  # 8 at t=2 is the same event as 9 at t=1
    assert pick_peaks(samples, 0, 2.0, "peak") == []


def test_start_time_comes_from_the_recording_name() -> None:
    assert start_from_name("B8A44FD52B7F_20260312_103023_ADA3.mkv") == datetime(2026, 3, 12, 10, 30, 23)
    assert start_from_name("20260801-061504.mp4") == datetime(2026, 8, 1, 6, 15, 4)
    assert start_from_name("clip.mkv") is None
    assert start_from_name("B8A44FD52B7F_20261399_250000.mkv") is None  # not a date


def test_start_time_parsing_and_clock() -> None:
    assert parse_start_time("16:20").strftime("%H:%M:%S") == "16:20:00"
    assert parse_start_time("2026-08-28 16:20:30") == datetime(2026, 8, 28, 16, 20, 30)
    with pytest.raises(ValueError):
        parse_start_time("half past four")
    assert clock_of(parse_start_time("23:59"), 90) == "00:00:30"
    assert clock_of(None, 3725) == "1:02:05"


def test_rolling_mean_keeps_the_length_and_the_edges() -> None:
    assert rolling_mean([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]
    out = rolling_mean([0.0, 0.0, 3.0, 0.0, 0.0], 3)
    assert len(out) == 5 and out[2] == pytest.approx(1.0) and out[0] == 0.0


def test_summary_and_csv(tmp_path) -> None:
    samples = [sample(i, float(i), count=c, peak=c / 10) for i, c in enumerate([10.0, 30.0, 20.0])]
    s = summarize(samples, datetime(2026, 1, 1, 8, 0, 0))
    assert s["samples"] == 3
    assert s["count"]["max"] == 30.0 and s["count"]["max_clock"] == "08:00:01"
    assert s["peak"]["mean"] == 2.0
    write_csv(tmp_path / "t.csv", samples, None, smooth=3)
    lines = (tmp_path / "t.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("sample,frame,seconds,clock,count,peak")
    assert lines[2].split(",")[3] == "0:00:01" and len(lines) == 4
