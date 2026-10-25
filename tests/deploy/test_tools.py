"""The deployment tools' logic, without cameras, devices or a display."""

import asyncio

import numpy as np
import pytest

from birdcount.deploy.tools import devices, discover_ids
from birdcount.deploy.tools.discover_ids import Probe
from birdcount.deploy.tools.mask_editor import Region, inside_polygon, normalized, ppm
from birdcount.deploy.topology import Camera


# --- discover_ids -------------------------------------------------------------


def test_mac_is_read_from_a_vapix_answer() -> None:
    assert discover_ids.parse_mac("root.Network.eth0.MACAddress=B8:A4:4F:D5:1C:3C\n") == "B8A44FD51C3C"
    assert discover_ids.parse_mac("root.Network.eth1.MACAddress = b8-a4-4f-d5-1c-87") == "B8A44FD51C87"
    assert discover_ids.parse_mac("Error: no such group") is None


def test_verify_flags_only_a_wrong_mac_as_failure(capsys) -> None:
    probes = [
        Probe("Z", Camera("10.0.0.1", "B8A44FD51C3C"), mac="B8A44FD51C3C"),
        Probe("Z", Camera("10.0.0.2"), mac="B8A44FD51C87"),
        Probe("Z", Camera("10.0.0.3"), error="timeout"),
    ]
    assert discover_ids.verify(probes) == 0
    probes.append(Probe("Z", Camera("10.0.0.4", "B8A44FD52AC4"), mac="B8A44FD52B7F"))
    assert discover_ids.verify(probes) == 1
    out = capsys.readouterr().out
    assert "1 correct, 1 WRONG, 1 unmapped, 1 unreachable" in out


# --- devices ------------------------------------------------------------------


class FakeDriver:
    def __init__(self, ip, on):
        self.ip, self.on, self.calls = ip, on, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def start(self):
        self.calls.append("start")
        self.on = True

    async def refresh(self):
        self.calls.append("refresh")

    async def is_on(self):
        return self.on

    async def stop(self):
        self.calls.append("stop")
        self.on = False


@pytest.mark.parametrize(
    ("action", "before", "outcome", "calls"),
    [
        ("status", True, "ON", []),
        ("status", None, "unknown", []),
        ("on", False, "-> ON", ["start", "refresh"]),
        ("off", True, "-> OFF", ["stop"]),
        ("toggle", True, "-> OFF", ["stop"]),
        ("toggle", False, "-> ON", ["start", "refresh"]),
    ],
)
def test_device_actions(action, before, outcome, calls) -> None:
    driver = FakeDriver("10.0.1.1", before)
    assert asyncio.run(devices.apply(driver, action, timeout=1.0)) == outcome
    assert driver.calls == calls


def test_toggle_refuses_an_unknown_state() -> None:
    with pytest.raises(RuntimeError, match="unknown"):
        asyncio.run(devices.apply(FakeDriver("10.0.1.1", None), "toggle", timeout=1.0))


def test_one_failing_device_fails_the_run_but_not_the_others(capsys) -> None:
    class Broken(FakeDriver):
        async def stop(self):
            raise OSError("unreachable")

    made = {}

    def factory(ip):
        made[ip] = (Broken if ip == "bad" else FakeDriver)(ip, True)
        return made[ip]

    code = asyncio.run(devices.run([("Z", "good"), ("Z", "bad")], factory, "off", timeout=1.0))
    assert code == 1 and made["good"].on is False
    assert "1/2 OK" in capsys.readouterr().out


# --- mask_editor ----------------------------------------------------------------


def test_rectangle_and_polygon_render_at_frame_size() -> None:
    rect = Region("rect", box=[30, 20, 10, 5])  # dragged up-left: normalised on render
    keep = rect.render((40, 30))
    assert keep.shape == (30, 40) and keep.sum() == 20 * 15 and keep[5:20, 10:30].all()

    poly = Region("poly", points=[[0, 0], [39, 0], [0, 29]])
    assert poly.render((40, 30)).sum() == 0  # not closed yet
    poly.closed = True
    keep = poly.render((40, 30))
    assert keep[1, 1] == 1 and keep[28, 38] == 0 and 550 < keep.sum() < 650


def test_a_loaded_mask_is_kept_and_resized_nearest() -> None:
    saved = np.zeros((30, 40), np.uint8)
    saved[:, :20] = 1
    region = Region("file", keep=saved)
    assert region.complete and region.render((40, 30)) is not saved
    assert np.array_equal(region.render((80, 60))[:, :40], np.ones((60, 40), np.uint8))


def test_geometry_helpers() -> None:
    assert normalized([5, 9, 1, 2]) == [1, 2, 5, 9]
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert inside_polygon(5, 5, square) and not inside_polygon(15, 5, square)
    frame = np.zeros((2, 3, 3), np.uint8)
    frame[..., 0] = 255  # blue in BGR
    data = ppm(frame)
    assert data.startswith(b"P6 3 2 255\n") and data[-3:] == b"\x00\x00\xff"  # RGB order
