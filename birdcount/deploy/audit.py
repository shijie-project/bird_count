"""Append-only JSONL audit trail of operator and alarm actions.

One instance per process, shared by every handler (v2 opened a handle per
handler on the same file). Thread-safe; with `path` None or unopenable,
`log()` is a cheap no-op.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._fh = None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = path.open("a", encoding="utf-8", buffering=1)
            except OSError as e:
                logger.error("Cannot open audit log %s: %s. Auditing is off.", path, e)

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def log(self, event: str, **fields: Any) -> None:
        """Append one record. Caller fields win over the built-in keys (replay / tests)."""
        if self._fh is None:
            return
        record = {
            "ts": time.time(),
            "iso": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            if self._fh is not None:
                self._fh.write(line)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


NULL_AUDIT = AuditLog(None)
