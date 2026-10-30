"""The recorder's archive on disk: which cameras, which days, and each day's clips in order.

The AXIS recorder export the farm's recordings come in:

    <root>/axisN/axis-<MAC>/<YYYYMMDD>/<HH>/<group>/<YYYYMMDD_HH>/<YYYYMMDD_HHMMSS>_<id>.mkv

A clip is ~5 minutes and its name is its local start time; a group is one
uninterrupted recording, and a new one starts after every interruption, so
the gaps between groups are hours nobody recorded. Only the clips count:
renders kept next to them (`*_overlay.mp4`) and folders that are not days
(`recording_groups`) are skipped.

`root` can be the whole archive, one recorder's folder (`axisN`) or one
camera's (`axis-<MAC>`).

stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_CAMERA_DIR = re.compile(r"^axis-([0-9A-Fa-f]{12})$")
_DAY_DIR = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_CLIP = re.compile(r"^(\d{8})_(\d{6})_[0-9A-Fa-f]{4}\.mkv$", re.IGNORECASE)
_SEARCH_DEPTH = 2  # root -> axisN -> axis-<MAC>


@dataclass(frozen=True)
class Clip:
    path: Path
    start: datetime  # local time, from the name


@dataclass(frozen=True)
class Camera:
    mac: str
    dirs: tuple[Path, ...]  # its axis-<MAC> folders: one, unless the export split it

    @property
    def group(self) -> str:
        """The recorder folder it was exported under (`axisN`)."""
        return ",".join(sorted({d.parent.name for d in self.dirs}))


def find_cameras(root: Path) -> dict[str, Camera]:
    """Every camera folder under `root`, by MAC."""
    found: dict[str, list[Path]] = {}
    level = [root]
    for _ in range(_SEARCH_DEPTH + 1):
        below = []
        for d in level:
            m = _CAMERA_DIR.match(d.name)
            if m:
                found.setdefault(m[1].upper(), []).append(d)
                continue
            try:
                below += sorted(p for p in d.iterdir() if p.is_dir())
            except OSError:  # a drive's `System Volume Information`
                continue
        level = below
    return {mac: Camera(mac, tuple(dirs)) for mac, dirs in sorted(found.items())}


def clip_start(path: Path) -> datetime | None:
    """A recorder clip's start, None for anything else (renders, notes, other videos)."""
    m = _CLIP.match(path.name)
    if m is None:
        return None
    try:
        return datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def days_of(camera: Camera) -> dict[str, list[Clip]]:
    """{YYYY-MM-DD: its clips in start order}, oldest day first.

    A clip belongs to the day it starts on; the same clip name found twice
    (an export copied over another) is counted once.
    """
    by_name: dict[str, Clip] = {}
    for d in camera.dirs:
        for day_dir in sorted(p for p in d.iterdir() if p.is_dir() and _DAY_DIR.match(p.name)):
            for path in day_dir.rglob("*.mkv"):
                start = clip_start(path)
                if start is not None:
                    by_name.setdefault(path.name.lower(), Clip(path, start))
    days: dict[str, list[Clip]] = {}
    for clip in sorted(by_name.values(), key=lambda c: (c.start, c.path.name)):
        days.setdefault(f"{clip.start:%Y-%m-%d}", []).append(clip)
    return days
