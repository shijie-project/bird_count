"""Typed settings, read from the project's single `.env`.

One `.env` at PROJECT_ROOT serves every app. Each app declares a `Settings`
subclass holding only the keys it reads; all of them share the conventions
below, so a key means the same thing wherever it is read:

* Relative paths are anchored at PROJECT_ROOT (`ProjectPath`), never at cwd.
* A blank optional path means "off" (`OptionalProjectPath` -> None).
* Values are validated on construction, so a bad `.env` fails at startup with
  the offending key named, not minutes later in a worker process.
* Settings objects are frozen: configuration does not change at runtime.
* Process environment variables override `.env`.

`.env.example` is the catalogue of every key. `Settings.load()` warns about
keys in `.env` that it does not list — v2 silently ignored unknown keys, so a
typo such as `MONITOR_MODE` for `ENABLE_MONITOR` simply did nothing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

from birdcount.core.paths import PROJECT_ROOT, resolve_path


logger = logging.getLogger(__name__)

ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"

# `KEY=value` or `export KEY=value`. Only key names are needed, so a full dotenv
# parser (quoting, interpolation) would be wasted here.
_ENV_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _blank_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _resolve_optional(path: Path | None) -> Path | None:
    return None if path is None else resolve_path(path)


ProjectPath = Annotated[Path, AfterValidator(resolve_path)]
OptionalProjectPath = Annotated[Path | None, BeforeValidator(_blank_to_none), AfterValidator(_resolve_optional)]


def env_file_keys(path: Path) -> set[str]:
    """Upper-cased key names assigned in a dotenv file (empty if it is missing)."""
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ENV_KEY_RE.match(line)
        if match:
            keys.add(match.group(1).upper())
    return keys


def unknown_env_keys(env_file: Path = ENV_FILE, example_file: Path = ENV_EXAMPLE_FILE) -> list[str]:
    """Keys set in `env_file` that `example_file` does not document, sorted."""
    return sorted(env_file_keys(env_file) - env_file_keys(example_file))


class Settings(BaseSettings):
    """Base for every app's settings. Subclasses only add fields."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # one .env serves every app; each reads only its own keys
        frozen=True,
    )

    @classmethod
    def load(cls) -> Self:
        """Construct from the environment + `.env`, warning about undocumented keys."""
        for key in unknown_env_keys():
            logger.warning("%s sets %s, which no setting reads (typo? see .env.example).", ENV_FILE.name, key)
        return cls()


class CoreSettings(Settings):
    """Keys every app reads."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: ProjectPath = Path("logs")
    data_root: ProjectPath = Path("../data")
    output_root: ProjectPath = Path("outputs")
