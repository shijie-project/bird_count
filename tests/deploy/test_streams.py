"""Which streams the runtime opens, how they are named, and which zone each belongs to."""

import pytest

from birdcount.deploy.settings import DeploySettings
from birdcount.deploy.sources import Source, redact
from birdcount.deploy.streams import StreamError, camera_uri, resolve_streams
from birdcount.deploy.topology import parse_topology


MAC_A, MAC_B = "B8A44FD51C3C", "B8A44FD51C87"
TOPOLOGY = parse_topology(
    {
        "zones": [
            {"name": "Z1", "cameras": [{"ip": "10.0.0.1", "mac": MAC_A}, "10.0.0.2"], "speakers": ["10.0.1.1"]},
            {"name": "Z2", "cameras": [{"ip": "10.0.0.3", "mac": MAC_B}], "smart_plugs": ["10.0.2.1"]},
        ]
    }
)


def settings(**kw) -> DeploySettings:
    return DeploySettings(**{"num_streams": None, **kw})


def test_camera_mode_is_every_camera_in_zone_order() -> None:
    streams = resolve_streams(settings(source_type="camera", camera_username="u", camera_password="p@ss"), TOPOLOGY)
    assert [(s.sid, s.label, s.mac, s.zone.name) for s in streams] == [
        (0, "10.0.0.1", MAC_A, "Z1"),
        (1, "10.0.0.2", None, "Z1"),
        (2, "10.0.0.3", MAC_B, "Z2"),
    ]
    assert streams[0].source.uri == "https://u:p%40ss@10.0.0.1/mjpg/1/video.mjpg"
    assert streams[0].source.display == "https://10.0.0.1/mjpg/1/video.mjpg"  # what gets logged
    assert streams[0].devices == {"10.0.1.1"} and streams[2].devices == {"10.0.2.1"}


def test_num_streams_keeps_the_first_n() -> None:
    assert [s.sid for s in resolve_streams(settings(source_type="camera", num_streams=2), TOPOLOGY)] == [0, 1]


def test_video_mode_matches_clips_to_zones_by_mac(tmp_path) -> None:
    for name in (f"{MAC_B}_20260101.mkv", "unknown_camera.mp4", f"{MAC_A}.avi", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    streams = resolve_streams(settings(source_type="video", video_paths=str(tmp_path)), TOPOLOGY)
    by_label = {s.label: s for s in streams}
    assert sorted(by_label) == sorted([f"{MAC_B}_20260101", "unknown_camera", MAC_A])  # .txt is not a clip
    assert by_label[f"{MAC_B}_20260101"].zone.name == "Z2" and by_label[MAC_A].zone.name == "Z1"
    assert by_label["unknown_camera"].zone is None and by_label["unknown_camera"].devices == frozenset()
    assert all(s.source.is_file for s in streams) and [s.sid for s in streams] == [0, 1, 2]


def test_video_paths_take_files_and_folders_and_reject_missing_ones(tmp_path) -> None:
    clip = tmp_path / "a.mkv"
    clip.write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mp4").write_bytes(b"")
    streams = resolve_streams(settings(source_type="video", video_paths=f"{clip}, {tmp_path / 'sub'}"), TOPOLOGY)
    assert [s.label for s in streams] == ["a", "b"]
    with pytest.raises(StreamError, match="does not exist"):
        resolve_streams(settings(source_type="video", video_paths=str(tmp_path / "nope")), TOPOLOGY)
    (tmp_path / "empty").mkdir()
    with pytest.raises(StreamError, match="no video files"):
        resolve_streams(settings(source_type="video", video_paths=str(tmp_path / "empty")), TOPOLOGY)


def test_uri_helpers() -> None:
    assert camera_uri("rtsp://{ip}/live", "1.2.3.4", "", "") == "rtsp://1.2.3.4/live"
    assert redact("https://a:b@host:8443/x?y=1") == "https://host:8443/x?y=1"
    assert Source(0, "C:/clips/a.mkv", True).display == "C:/clips/a.mkv"


def test_camera_url_needs_the_ip_placeholder() -> None:
    with pytest.raises(ValueError, match="ip"):
        settings(camera_url="https://camera/video")
