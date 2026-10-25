"""Where the project lives on disk, and what relative paths are relative to.

Every relative path the project reads — from `.env`, a CLI flag or a config
file — is resolved against `PROJECT_ROOT`, never against the current working
directory. (v2 resolved them against cwd, so `../data` pointed somewhere
different depending on which directory a script was launched from.)
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    # BIRDCOUNT_ROOT exists for non-editable installs, where __file__ lives in
    # site-packages rather than in the checkout.
    override = os.environ.get("BIRDCOUNT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _find_project_root()


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Absolute paths pass through; relative ones are anchored at PROJECT_ROOT."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()
