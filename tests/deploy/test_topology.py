"""configs/topology.yaml: parsing, the strict checks, and writing MACs back."""

from pathlib import Path

import pytest

from birdcount.core.paths import PROJECT_ROOT
from birdcount.deploy.topology import Camera, TopologyError, load_topology, parse_topology, set_camera_macs


def zone(**kw) -> dict:
    return {"name": "Z1", "cameras": ["10.0.0.1"], **kw}


def test_the_shipped_topology_loads() -> None:
    topology = load_topology(PROJECT_ROOT / "configs" / "topology.yaml")
    assert len(topology.zones) >= 1 and any(True for _ in topology.cameras())


def test_cameras_with_and_without_a_mac() -> None:
    topology = parse_topology(
        {
            "zones": [
                zone(
                    cameras=["10.0.0.1", {"ip": "10.0.0.2", "mac": "b8:a4:4f:d5:1c:3c"}],
                    speakers=["10.0.1.1"],
                    smart_plugs=["10.0.2.1"],
                ),
                {"name": "Z2", "cameras": [{"ip": "10.0.0.3", "mac": ""}]},
            ]
        }
    )
    z1, z2 = topology.zones
    assert z1.cameras == (Camera("10.0.0.1"), Camera("10.0.0.2", "B8A44FD51C3C"))
    assert z2.cameras == (Camera("10.0.0.3"),) and z2.speakers == () and z2.devices == frozenset()
    assert z1.devices == {"10.0.1.1", "10.0.2.1"}
    assert topology.zone_of_mac("B8A44FD51C3C") is z1
    assert topology.zone_of_mac("000000000000") is None and topology.zone_of_mac(None) is None


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "zones"),
        ({"zones": [zone()], "extra": 1}, "top-level"),
        ({"zones": [{"cameras": []}]}, "name"),
        ({"zones": [zone(camera_ids=["B8A44FD51C3C"])]}, "camera_ids"),
        ({"zones": [zone(thresholds={"warn": 1})]}, "alarm.json"),
        ({"zones": [zone(speaker=["10.0.1.1"])]}, "unknown key"),
        ({"zones": [zone(cameras=[{"ip": "10.0.0.1", "mac": "nope"}])]}, "not a MAC"),
        ({"zones": [zone(cameras=["10.0.0.1/24"])]}, "not an IP"),
        ({"zones": [zone(), zone(name="Z2")]}, "camera IP used more than once"),
        ({"zones": [zone(), zone(name="Z1", cameras=["10.0.0.2"])]}, "zone name"),
        (
            {
                "zones": [
                    zone(cameras=[{"ip": "10.0.0.1", "mac": "B8A44FD51C3C"}]),
                    zone(name="Z2", cameras=[{"ip": "10.0.0.2", "mac": "B8A44FD51C3C"}]),
                ]
            },
            "camera MAC",
        ),
    ],
)
def test_mistakes_fail_the_load_with_the_reason(raw, message) -> None:
    with pytest.raises(TopologyError, match=message):
        parse_topology(raw)


def test_writing_macs_keeps_comments_and_layout(tmp_path: Path) -> None:
    text = (
        "# farm layout\n"
        "zones:\n"
        "  - name: Z1  # front\n"
        "    cameras:\n"
        '      - "10.0.0.1"   # left door\n'
        "      - {ip: 10.0.0.2, mac: 000000000000}\n"
        "      - 10.0.0.3\n"
    )
    new, missing = set_camera_macs(text, {"10.0.0.1": "B8A44FD51C3C", "10.0.0.2": "B8A44FD51C87", "10.0.0.9": "X"})
    assert missing == ["10.0.0.9"]
    assert '      - {ip: "10.0.0.1", mac: "B8A44FD51C3C"}   # left door\n' in new
    assert '      - {ip: "10.0.0.2", mac: "B8A44FD51C87"}\n' in new
    assert "# farm layout\n" in new and "      - 10.0.0.3\n" in new
    path = tmp_path / "t.yaml"
    path.write_text(new, encoding="utf-8")
    cams = {c.ip: c.mac for _, c in load_topology(path).cameras()}
    assert cams == {"10.0.0.1": "B8A44FD51C3C", "10.0.0.2": "B8A44FD51C87", "10.0.0.3": None}
