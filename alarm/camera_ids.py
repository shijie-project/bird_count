"""The one translation point between our bare-MAC identity and the alarm config.

This project keys every per-camera asset on the bare MAC (see
`runtime.camera_identity`). The vendored alarm config addresses the same
cameras as `axisN/MAC`, where `axisN` is the recorder group they were delivered
under:

    B8A44FD51C3C          our identity, from topology.yaml `camera_ids:`
    axis1/B8A44FD51C3C    the key in configs/alarm.json

`configs/alarm.json` is left in the vendor's spelling deliberately: those 16
thresholds are their calibration, the id is quoted verbatim in the SMS body
(`Camera ID: axis1/B8A44FD51C3C`), and rewriting the file would cost the
traceability for no gain. So the prefix is added back here, once, instead.

The MAC already identifies the camera on its own, so the mapping is a suffix
lookup — but only a *unique* one is accepted. Two cameras sharing a MAC under
different recorders should not happen; if it ever did, guessing would attach
the wrong threshold to a live camera, so this returns None and lets the caller
report the stream as uncovered.
"""

from __future__ import annotations

from collections.abc import Iterable


def normalize_camera_id(camera_id: str) -> str:
    return camera_id.replace("\\", "/").strip("/")


def mac_of(camera_id: str) -> str:
    """The MAC half of an `axisN/MAC` id (or the id itself when already bare)."""
    return normalize_camera_id(camera_id).rsplit("/", 1)[-1].upper()


def resolve_camera_id(*, mac: str | None, known: Iterable[str]) -> str | None:
    """Bare MAC -> the matching camera id in the alarm config, or None.

    None means "this camera is not in the alarm config" — a legitimate state
    (the topology has 21 cameras, the alarm config covers 16), reported once at
    startup rather than raised.
    """
    if not mac:
        return None

    known = list(known)
    target = mac.upper()

    # An alarm config keyed on bare MACs needs no translation at all.
    if target in known:
        return target

    matches = [cid for cid in known if mac_of(cid) == target]
    return matches[0] if len(matches) == 1 else None
