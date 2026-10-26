from pathlib import Path

import pytest
from pydantic import ValidationError

from birdcount.core.paths import PROJECT_ROOT
from birdcount.core.settings import CoreSettings, OptionalProjectPath, Settings, unknown_env_keys


@pytest.fixture(autouse=True)
def _clean_environ(monkeypatch):
    """Keep the developer's shell environment out of every test."""
    for key in CoreSettings.model_fields:
        monkeypatch.delenv(key.upper(), raising=False)
    monkeypatch.delenv("MASK_DIR", raising=False)


def _env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_resolve_against_project_root(tmp_path) -> None:
    s = CoreSettings(_env_file=_env(tmp_path, ""))
    assert s.log_level == "INFO"
    assert s.data_root == (PROJECT_ROOT.parent / "data").resolve()
    assert s.output_root == PROJECT_ROOT / "outputs"


def test_env_file_paths_are_project_relative(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    s = CoreSettings(_env_file=_env(tmp_path, "OUTPUT_ROOT=runs/a\n"))
    assert s.output_root == PROJECT_ROOT / "runs" / "a"


def test_process_environment_overrides_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = CoreSettings(_env_file=_env(tmp_path, "LOG_LEVEL=WARNING\n"))
    assert s.log_level == "DEBUG"


def test_invalid_value_fails_naming_the_key(tmp_path) -> None:
    with pytest.raises(ValidationError, match="log_level"):
        CoreSettings(_env_file=_env(tmp_path, "LOG_LEVEL=verbose\n"))


def test_settings_are_frozen(tmp_path) -> None:
    s = CoreSettings(_env_file=_env(tmp_path, ""))
    with pytest.raises(ValidationError):
        s.log_level = "DEBUG"


class _WithOptionalPath(Settings):
    mask_dir: OptionalProjectPath = None


@pytest.mark.parametrize(("raw", "expected"), [("", None), ("   ", None), ("masks", PROJECT_ROOT / "masks")])
def test_blank_optional_path_means_off(tmp_path, raw, expected) -> None:
    s = _WithOptionalPath(_env_file=_env(tmp_path, f"MASK_DIR={raw}\n"))
    assert s.mask_dir == expected


def test_unknown_keys_are_reported(tmp_path) -> None:
    example = tmp_path / ".env.example"
    example.write_text("# comment\nLOG_LEVEL=INFO\nDATA_ROOT=../data\n", encoding="utf-8")
    env = _env(tmp_path, "log_level=DEBUG\nexport DATA_ROOT=x\nMONITOR_MODE=0\n# STALE=1\n")
    assert unknown_env_keys(env, example) == ["MONITOR_MODE"]


def test_missing_env_file_has_no_unknown_keys(tmp_path) -> None:
    assert unknown_env_keys(tmp_path / "absent", tmp_path / "absent.example") == []
