from pathlib import Path

from birdcount.core.paths import PROJECT_ROOT, resolve_path


def test_project_root_is_the_checkout() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "birdcount" / "__init__.py").is_file()


def test_relative_paths_ignore_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_path("../data") == (PROJECT_ROOT.parent / "data").resolve()
    assert resolve_path(Path("outputs/x")) == PROJECT_ROOT / "outputs" / "x"


def test_absolute_paths_pass_through(tmp_path) -> None:
    assert resolve_path(tmp_path / "a") == (tmp_path / "a").resolve()
