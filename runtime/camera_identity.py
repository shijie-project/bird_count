"""Bare-MAC camera identity, shared by the mask loader and the SMS alarm.

A camera is identified across this project by its **bare MAC** — the 12 hex
digits of the AXIS unit's hardware address, uppercase, no separators:

    B8A44FD51C3C          <- canonical
    axis1/B8A44FD51C3C    <- the pile-up alarm config's form (see alarm/camera_ids.py)
    mask_axis1_B8A44FD51C3C.png

The MAC is burned into the camera, so it survives re-IPing and re-cabling; the
`axisN` prefix is only a recorder-group label and carries no identity. Every
per-camera asset in this repo (topology `camera_ids:`, mask filenames) is keyed
on the bare MAC, and `alarm/camera_ids.py` translates to the vendor's
`axisN/MAC` form at that one boundary.

`extract_mac` therefore accepts any of the shapes above: it pulls the MAC out of
whatever string it is handed, provided the answer is unambiguous.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


# A run of exactly 12 hex digits, not glued to more alphanumerics on either
# side. The delimiters in practice are `_`, `-`, `/`, `.` and string ends.
#
# Note this also matches a 12-digit decimal string, so a video named
# `202607161200.mkv` would parse as a "MAC". Harmless: every caller checks the
# result against a known set (the alarm config's cameras, the mask directory
# index), and an unknown MAC degrades to "no coverage" with a startup warning.
_MAC_RE = re.compile(r"(?<![0-9A-Za-z])([0-9A-Fa-f]{12})(?![0-9A-Za-z])")


def normalize_mac(value: str) -> str:
    """Canonical form: uppercase, separators stripped.

    Accepts the usual human spellings (`b8:a4:4f:d5:1c:3c`, `B8-A4-4F-D5-1C-3C`)
    as well as an already-bare MAC. Returns "" if the result isn't 12 hex digits.
    """
    stripped = re.sub(r"[^0-9A-Za-z]", "", value or "").upper()
    return stripped if re.fullmatch(r"[0-9A-F]{12}", stripped) else ""


def extract_mac(text: str | Path) -> str | None:
    """Pull the bare MAC out of a filename, path or id. None when ambiguous.

    Ambiguity is deliberately an error rather than a guess: a string carrying
    two different MACs (say a path with both a folder and a file MAC that
    disagree) means the layout is not what we assumed, and silently picking one
    would attach the wrong threshold or the wrong mask to a camera.
    """
    if not text:
        return None
    found = {m.group(1).upper() for m in _MAC_RE.finditer(str(text))}
    return found.pop() if len(found) == 1 else None


def resolve_stream_mac(*, explicit: str | None, source: str, identifier: str) -> str | None:
    """Best-effort stream -> bare MAC, first hit wins.

    1. `camera_ids:` in topology.yaml — authoritative when present (camera mode);
    2. the stream source, which in video mode is the clip path and carries the
       MAC either as the filename (`B8A44FD51C3C.mkv`) or in the delivery
       package's `.../axisN/axis-<MAC>/clip.mkv` layout;
    3. the stream identifier (video stem, or a camera IP — an IP yields None).

    None means "this stream has no known camera identity", which every caller
    treats as a degraded-but-running state, never a failure.
    """
    if explicit:
        mac = normalize_mac(explicit) or extract_mac(explicit)
        if mac:
            return mac

    for candidate in (source, identifier):
        mac = extract_mac(candidate)
        if mac:
            return mac

    return None


def index_by_mac(paths: Iterable[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Index files by the MAC in their name.

    Returns `(unique, ambiguous)`. A MAC claimed by more than one file lands in
    `ambiguous` and is left out of `unique` — the caller reports it rather than
    picking arbitrarily. An exact `<MAC>.png` always wins over a decorated name
    (`mask_axis1_<MAC>.png`), so the canonical spelling is never the ambiguous one.
    """
    by_mac: dict[str, list[Path]] = {}
    for path in paths:
        mac = extract_mac(path.stem)
        if mac:
            by_mac.setdefault(mac, []).append(path)

    unique: dict[str, Path] = {}
    ambiguous: dict[str, list[Path]] = {}
    for mac, matches in by_mac.items():
        if len(matches) == 1:
            unique[mac] = matches[0]
            continue
        exact = [p for p in matches if p.stem.upper() == mac]
        if len(exact) == 1:
            unique[mac] = exact[0]
        else:
            ambiguous[mac] = sorted(matches)
    return unique, ambiguous
