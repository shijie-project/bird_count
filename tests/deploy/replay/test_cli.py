import pytest

from birdcount.deploy.replay.__main__ import main, parse_days


def test_days_and_ranges() -> None:
    assert parse_days("2026-03-11") == {"2026-03-11"}
    assert parse_days("2026-03-30..2026-04-01, 2026-03-11") == {"2026-03-30", "2026-03-31", "2026-04-01", "2026-03-11"}
    with pytest.raises(ValueError):
        parse_days("2026-03-12..2026-03-11")


def test_a_camera_without_recordings_is_refused(tmp_path, capsys) -> None:
    (tmp_path / "axis1" / "axis-B8A44FD51C3C" / "20260311").mkdir(parents=True)
    assert main([str(tmp_path), "--list"]) == 0
    assert "B8A44FD51C3C" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--camera", "E82725003277"])
    assert "no recordings of E82725003277" in capsys.readouterr().err
