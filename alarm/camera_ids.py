"""Resolve a runtime stream to the `axisN/MAC` camera id the alarm config uses.

The alarm config addresses cameras as `axis1/B8A44FD52B7F` (site + camera MAC),
while `runtime.config` addresses streams by index and by `identifier` (a camera
IP in camera mode, a video file stem in video mode). This module holds the
mapping rules that bridge the two.

Resolution order (first hit wins) is implemented by `resolve_camera_id`:

1. an explicit `camera_ids:` entry in topology.yaml — always authoritative;
2. the stream source path, when it follows the `.../axisN/axis-<MAC>/clip.mkv`
   layout the delivery video samples use;
3. the stream identifier, when it is already a valid camera id.

Streams that resolve to nothing simply get no alarm coverage — the handler logs
them once at startup rather than failing the run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


_AXIS_RE = re.compile(r"axis[1-9][0-9]*", re.IGNORECASE)


def normalize_camera_id(camera_id: str) -> str:
    return camera_id.replace("\\", "/").strip("/")


def camera_id_from_path(path: str | Path) -> str | None:
    """Parse `axisN/MAC` out of a `.../axisN/axis-<MAC>/clip.mkv` style path.

    Returns None when the path doesn't carry both halves.
    """
    axis = None
    camera = None
    for part in Path(path).parts:
        lower = part.lower()
        if _AXIS_RE.fullmatch(lower):
            axis = lower
        if lower.startswith("axis-"):
            camera = part.split("axis-", 1)[1]
    if axis and camera:
        return f"{axis}/{camera}"
    return None


def resolve_camera_id(
    *,
    explicit: str | None,
    source: str,
    identifier: str,
    known: Iterable[str],
) -> str | None:
    """Best-effort stream -> alarm camera id. See module docstring for the order.

    `known` is the camera id set from the alarm config; a candidate that isn't
    in it is rejected, so a typo degrades to "no alarm coverage" (visible in the
    startup log) instead of a KeyError deep inside the state machine.
    """
    known = set(known)

    if explicit:
        candidate = normalize_camera_id(explicit)
        return candidate if candidate in known else None

    from_path = camera_id_from_path(source)
    if from_path and from_path in known:
        return from_path

    candidate = normalize_camera_id(identifier)
    if candidate in known:
        return candidate

    return None
