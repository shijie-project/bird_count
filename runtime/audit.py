import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


class AuditLog:
    """
    Append-only JSONL audit log.

    Thread-safe within a process; cross-process append is also safe because
    each emitted record fits well under PIPE_BUF on every supported platform
    and the file is opened in O_APPEND mode (each write atomically seeks-end).

    Disabled when `path` is falsy or the file can't be opened — `log()` then
    becomes a cheap no-op. Disabled-ness is derived from `self._fh`, so there's
    no separate flag to keep in sync.
    """

    def __init__(self, path: Optional[str], name: str = "audit"):
        self.name = name
        self.path: Optional[Path] = Path(path) if path else None
        self._lock = threading.Lock()
        self._fh = self._open() if self.path is not None else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(self.path, "a", encoding="utf-8", buffering=1)
            logger.info("[%s] Audit log opened: %s", self.name, self.path)
            return fh
        except Exception as e:
            logger.error("[%s] Failed to open audit log at %s: %s", self.name, self.path, e, exc_info=True)
            return None

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            with self._lock:
                self._fh.close()
        except Exception as e:
            logger.debug("[%s] close failed: %s", self.name, e)
        finally:
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True iff the audit log is open and accepting writes."""
        return self._fh is not None

    def log(self, event: str, **fields: Any) -> None:
        """
        Append one JSON record. No-op when the log is disabled.

        Caller-supplied `fields` win on key collisions with the built-in
        keys (`ts`, `iso`, `pid`, `event`) — intentional, lets callers
        override timestamps for testing / replay.
        """
        if self._fh is None:
            return
        record = {
            "ts": time.time(),
            "iso": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        try:
            line = json.dumps(record, default=str)
            with self._lock:
                # Re-check inside the lock — close() may have run between
                # the early-out above and lock acquisition.
                if self._fh is None:
                    return
                self._fh.write(line + "\n")
        except Exception as e:
            logger.debug("[%s] audit write failed: %s", self.name, e)
