"""Copying clips off a slow disk one at a time, ahead of the camera-days decoding them.

The recordings sit on a USB hard disk. Read by several decoders at once it
seeks between them: measured 31 MB/s with the disk 98 % busy for six
camera-days, against 97 MB/s copying one file after the other. So one thread
copies the clips each camera-day will need next to a local folder, whole
files in turn, a few clips ahead of it, and the decoders read the local
copies; a copy is deleted once its clip is counted. The local folder holds at
most `ahead` clips per camera-day running (~110 MB each).

The copier serves the camera-day with the fewest clips ready first, so none
waits while another runs far ahead.

stdlib only.
"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path


logger = logging.getLogger(__name__)

_WAIT = 0.2  # s between checks of a stop flag while waiting


class Stopped(Exception):
    """Waiting for a clip was given up: the replay is stopping."""


class InPlace:
    """Clips read where they are (already on a fast disk)."""

    def __init__(self, sources: Sequence[Path]):
        self.sources = list(sources)

    def fetch(self, i: int, stop: threading.Event | None = None) -> Path:
        return self.sources[i]

    def done(self, i: int) -> None:
        pass

    def close(self) -> None:
        pass


class _Plan:
    """One camera-day's clips, in the order it decodes them, and how far it has got."""

    def __init__(self, cache: ClipCache, key: int, sources: Sequence[Path]):
        self.cache, self.key, self.sources = cache, key, list(sources)
        self.pos = 0  # the clip it decodes now or next
        self.local: dict[int, Path] = {}  # copied, not yet done
        self.failed: dict[int, str] = {}
        self.busy: int | None = None  # being copied

    def wanted(self, ahead: int) -> int | None:
        for i in range(self.pos, min(self.pos + ahead, len(self.sources))):
            if i not in self.local and i not in self.failed and i != self.busy:
                return i
        return None

    def ready_ahead(self) -> int:
        return sum(1 for i in self.local if i >= self.pos)

    def fetch(self, i: int, stop: threading.Event | None = None) -> Path:
        """The local copy of clip `i`, once copied; OSError if it could not be."""
        with self.cache._cond:
            self.pos = i
            self.cache._cond.notify_all()
            while i not in self.local and i not in self.failed:
                if self.cache._closed or (stop is not None and stop.is_set()):
                    raise Stopped
                self.cache._cond.wait(_WAIT)
            if i in self.failed:
                raise OSError(self.failed[i])
            return self.local[i]

    def done(self, i: int) -> None:
        with self.cache._cond:
            path = self.local.pop(i, None)
            self.pos = max(self.pos, i + 1)
            self.cache._cond.notify_all()
        if path is not None:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        with self.cache._cond:
            self.cache._plans.pop(self.key, None)
            leftovers, self.local = list(self.local.values()), {}
            self.cache._cond.notify_all()
        for path in leftovers:
            path.unlink(missing_ok=True)


class ClipCache:
    def __init__(self, root: Path, *, ahead: int = 3):
        if ahead < 1:
            raise ValueError("ahead must be at least 1")
        self.root, self.ahead = root, ahead
        shutil.rmtree(root, ignore_errors=True)  # a stopped replay's copies
        root.mkdir(parents=True)
        self._cond = threading.Condition()
        self._plans: dict[int, _Plan] = {}
        self._next_key = 0
        self._closed = False
        self.copied_bytes = 0
        self._thread = threading.Thread(target=self._copy_loop, name="ClipCopier", daemon=True)
        self._thread.start()

    def open(self, sources: Sequence[Path]) -> _Plan:
        """Start copying a camera-day's clips, in the order it will ask for them."""
        with self._cond:
            key, self._next_key = self._next_key, self._next_key + 1
            plan = self._plans[key] = _Plan(self, key, sources)
            self._cond.notify_all()
            return plan

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join()
        shutil.rmtree(self.root, ignore_errors=True)

    def _pick(self) -> tuple[_Plan, int] | None:
        """The next clip to copy: for the camera-day with the fewest clips ready ahead of it."""
        best = None
        for plan in self._plans.values():
            i = plan.wanted(self.ahead)
            if i is not None and (best is None or plan.ready_ahead() < best[0].ready_ahead()):
                best = (plan, i)
        return best

    def _copy_loop(self) -> None:
        while True:
            with self._cond:
                while not self._closed and (job := self._pick()) is None:
                    self._cond.wait()
                if self._closed:
                    return
                plan, i = job
                plan.busy = i
                src = plan.sources[i]
                dst = self.root / f"{plan.key}_{i}{src.suffix}"
            error, size = None, 0
            tmp = dst.with_name(f"{dst.name}.part")
            try:
                shutil.copyfile(src, tmp)
                tmp.replace(dst)
                size = dst.stat().st_size
            except OSError as e:
                tmp.unlink(missing_ok=True)
                error = f"cannot copy {src}: {e}"
                logger.warning("%s", error)
            with self._cond:
                plan.busy = None
                if self._plans.get(plan.key) is not plan:  # closed while copying
                    dst.unlink(missing_ok=True)
                elif error is None:
                    plan.local[i] = dst
                    self.copied_bytes += size
                else:
                    plan.failed[i] = error
                self._cond.notify_all()
