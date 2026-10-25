"""What the runtime will capture: one `Stream` per camera, numbered by sid.

Camera mode: every camera in the topology, in zone order. Video mode: one
stream per clip in VIDEO_PATHS; a clip takes the zone of the topology camera
whose MAC is in its filename — by MAC, never by position, since the i-th clip
in a folder says nothing about the i-th camera — so its alarm drives the
devices of the pen it was recorded in. A clip from a camera the topology
does not list runs with no zone (it alarms, but switches nothing on).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from birdcount.core.camera_identity import resolve_stream_mac
from birdcount.core.paths import resolve_path
from birdcount.deploy.settings import DeploySettings
from birdcount.deploy.sources import VIDEO_SUFFIXES, Source
from birdcount.deploy.topology import Topology, Zone


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stream:
    sid: int
    source: Source
    label: str  # camera IP or clip name: what operators recognise
    mac: str | None
    zone: Zone | None

    @property
    def devices(self) -> frozenset[str]:
        return self.zone.devices if self.zone is not None else frozenset()


class StreamError(ValueError):
    pass


def resolve_streams(settings: DeploySettings, topology: Topology) -> list[Stream]:
    if settings.source_type == "video":
        streams = _video_streams(settings.video_paths, topology)
    else:
        streams = _camera_streams(settings, topology)
    if not streams:
        raise StreamError(
            "no streams: the topology lists no cameras"
            if settings.source_type == "camera"
            else f"no video files found in VIDEO_PATHS={settings.video_paths!r}"
        )
    if settings.num_streams is not None and settings.num_streams < len(streams):
        logger.info(
            "NUM_STREAMS=%d: using %d of %d streams.", settings.num_streams, settings.num_streams, len(streams)
        )
        streams = streams[: settings.num_streams]
    return streams


def camera_uri(template: str, ip: str, username: str, password: str) -> str:
    uri = template.format(ip=ip)
    if not username:
        return uri
    scheme, sep, rest = uri.partition("://")
    if not sep:
        return uri
    return f"{scheme}://{quote(username, safe='')}:{quote(password, safe='')}@{rest}"


def _camera_streams(settings: DeploySettings, topology: Topology) -> list[Stream]:
    password = settings.camera_password.get_secret_value()
    return [
        Stream(
            sid=sid,
            source=Source(sid, camera_uri(settings.camera_url, cam.ip, settings.camera_username, password), False),
            label=cam.ip,
            mac=cam.mac,
            zone=zone,
        )
        for sid, (zone, cam) in enumerate(topology.cameras())
    ]


def video_files(spec: str) -> list[Path]:
    """Files and folders (every clip inside, sorted), comma-separated; relative to the project."""
    files: list[Path] = []
    for entry in (e.strip() for e in spec.split(",")):
        if not entry:
            continue
        path = resolve_path(entry)
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            raise StreamError(f"VIDEO_PATHS entry {entry!r} does not exist ({path})")
    return files


def _video_streams(spec: str, topology: Topology) -> list[Stream]:
    streams = []
    for sid, path in enumerate(video_files(spec)):
        mac = resolve_stream_mac(explicit=None, source=str(path), identifier=path.stem)
        streams.append(Stream(sid, Source(sid, str(path), True), path.stem, mac, topology.zone_of_mac(mac)))
    return streams


def describe(streams: list[Stream]) -> list[str]:
    """One line per stream for the startup log and `--check`."""
    return [f"#{s.sid:<3} {s.label:<24} {s.mac or '(no MAC)':<14} {s.zone.name if s.zone else '-'}" for s in streams]
