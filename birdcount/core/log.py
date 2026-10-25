"""Process-wide logging setup. Stdlib only.

v2's `setup_logging` lived in a module that imported torch, so every process
that merely configured logging (the frame grabber, the monitor) paid ~500 MB
of RSS and ~1.7 s of startup for a library it never used. Keep this module
dependency-free.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(processName)s:%(process)d | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Libraries that log every request or decode at INFO.
_NOISY_LOGGERS = ("httpcore", "httpx", "matplotlib", "PIL", "urllib3")


def setup_logging(level: str | int = "INFO", *, log_file: Path | None = None) -> None:
    """Configure the root logger for the calling process.

    Call once per process, including spawned children: with the `spawn` start
    method a child does not inherit its parent's handlers. Calling again
    replaces the previous configuration rather than stacking handlers.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATE_FORMAT, handlers=handlers, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
