"""The farm layout: zones, each with cameras, speakers and smart plugs.

The single reader of `configs/topology.yaml` — the runtime, the camera-id
discovery tool and the device tool all load it here (v2 had three parsers).

    zones:
      - name: Zone_01_Right_Front
        cameras:
          - {ip: 138.25.209.125, mac: B8A44FD51C3C}
          - 138.25.209.122            # MAC not known yet
        speakers: [138.25.209.236]
        smart_plugs: [138.25.209.31]

A camera's MAC sits next to its IP (v2 kept a separate `camera_ids:` list that
had to stay positionally aligned with `cameras:`). The MAC keys the camera's
region mask and alarm threshold; a camera without one runs unmasked and never
alarms, which `--check` reports.

Loading is strict: an unknown key, a malformed IP or MAC, or an IP / MAC /
zone name used twice is an error naming the zone, not a camera silently
running without its mask.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from birdcount.core.camera_identity import normalize_mac


class TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class Camera:
    ip: str
    mac: str | None = None


@dataclass(frozen=True)
class Zone:
    name: str
    cameras: tuple[Camera, ...] = ()
    speakers: tuple[str, ...] = ()
    smart_plugs: tuple[str, ...] = ()

    @property
    def devices(self) -> frozenset[str]:
        return frozenset(self.speakers) | frozenset(self.smart_plugs)


@dataclass(frozen=True)
class Topology:
    zones: tuple[Zone, ...]
    path: Path | None = field(default=None, compare=False)

    def cameras(self) -> Iterator[tuple[Zone, Camera]]:
        for zone in self.zones:
            for camera in zone.cameras:
                yield zone, camera

    def zone_of_mac(self, mac: str | None) -> Zone | None:
        if not mac:
            return None
        return next((zone for zone, cam in self.cameras() if cam.mac == mac), None)


_ZONE_KEYS = {"name", "cameras", "speakers", "smart_plugs"}
_CAMERA_KEYS = {"ip", "mac"}


def load_topology(path: Path) -> Topology:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as e:
        raise TopologyError(f"cannot read {path}: {e}") from e
    except yaml.YAMLError as e:
        raise TopologyError(f"{path} is not valid YAML: {e}") from e
    try:
        return parse_topology(raw, path)
    except TopologyError as e:
        raise TopologyError(f"{path}: {e}") from None


def parse_topology(raw: dict, path: Path | None = None) -> Topology:
    if not isinstance(raw, dict) or not isinstance(raw.get("zones"), list) or not raw["zones"]:
        raise TopologyError("needs a non-empty `zones:` list")
    extra = set(raw) - {"zones"}
    if extra:
        raise TopologyError(f"unknown top-level key(s) {sorted(extra)}")
    zones = tuple(_zone(z, i) for i, z in enumerate(raw["zones"]))
    _unique("zone name", [z.name for z in zones])
    _unique("camera IP", [c.ip for z in zones for c in z.cameras])
    _unique("camera MAC", [c.mac for z in zones for c in z.cameras if c.mac])
    _unique("device IP", [ip for z in zones for ip in (*z.speakers, *z.smart_plugs)])
    return Topology(zones, path)


def _zone(raw, index: int) -> Zone:
    if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
        raise TopologyError(f"zone #{index + 1} needs a `name:`")
    name = str(raw["name"]).strip()
    if "camera_ids" in raw:
        raise TopologyError(
            f"zone {name!r}: `camera_ids:` is gone; put each MAC next to its camera, as {{ip: ..., mac: ...}}"
        )
    if "thresholds" in raw:
        raise TopologyError(f"zone {name!r}: `thresholds:` is gone; thresholds live in configs/alarm.json")
    extra = set(raw) - _ZONE_KEYS
    if extra:
        raise TopologyError(f"zone {name!r}: unknown key(s) {sorted(extra)}")
    try:
        return Zone(
            name=name,
            cameras=tuple(_camera(c) for c in _list(raw, "cameras")),
            speakers=tuple(_ip(d) for d in _list(raw, "speakers")),
            smart_plugs=tuple(_ip(d) for d in _list(raw, "smart_plugs")),
        )
    except TopologyError as e:
        raise TopologyError(f"zone {name!r}: {e}") from None


def _list(raw: dict, key: str) -> list:
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TopologyError(f"`{key}:` must be a list")
    return value


def _camera(raw) -> Camera:
    if isinstance(raw, dict):
        extra = set(raw) - _CAMERA_KEYS
        if extra or "ip" not in raw:
            raise TopologyError(f"camera {raw!r}: expected {{ip: ..., mac: ...}}")
        ip, mac_raw = _ip(raw["ip"]), str(raw.get("mac") or "").strip()
        if not mac_raw:
            return Camera(ip)
        mac = normalize_mac(mac_raw)
        if not mac:
            raise TopologyError(f"camera {ip}: {mac_raw!r} is not a MAC (12 hex digits, e.g. B8A44FD51C3C)")
        return Camera(ip, mac)
    return Camera(_ip(raw))


def _ip(raw) -> str:
    text = str(raw).strip()
    try:
        ipaddress.ip_address(text)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9.-]+", text):  # a host name is fine too
            raise TopologyError(f"{text!r} is not an IP address or host name") from None
    return text


def _unique(what: str, values: list[str]) -> None:
    seen, dupes = set(), set()
    for v in values:
        (dupes if v in seen else seen).add(v)
    if dupes:
        raise TopologyError(f"{what} used more than once: {sorted(dupes)}")


# ----------------------------------------------------------------------------
# Writing camera MACs back (tools.discover_ids --write)
# ----------------------------------------------------------------------------


def set_camera_macs(text: str, macs: dict[str, str]) -> tuple[str, list[str]]:
    """Rewrite the camera entries of `macs` (ip -> MAC) in topology YAML text.

    Line-local, so comments and layout survive (a YAML round-trip would drop
    them). Returns the new text and the IPs that were not found as a list
    entry of their own.
    """
    lines = text.splitlines(keepends=True)
    missing = []
    for ip, mac in macs.items():
        q = re.escape(ip)
        entry = re.compile(
            rf"^(?P<lead>\s*-\s*)(?:[\"']?{q}[\"']?|\{{\s*ip:\s*[\"']?{q}[\"']?\s*(?:,\s*mac:\s*[^}}]*)?\}})"
            rf"(?P<tail>\s*(?:#.*)?)$"
        )
        for i, line in enumerate(lines):
            body = line.rstrip("\r\n")
            match = entry.match(body)
            if match:
                newline = line[len(body) :]
                lines[i] = f'{match["lead"]}{{ip: "{ip}", mac: "{mac}"}}{match["tail"]}{newline}'
                break
        else:
            missing.append(ip)
    return "".join(lines), missing
