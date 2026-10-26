"""Ask every camera in the topology for its MAC, over AXIS VAPIX.

The MAC is what keys a camera's region mask and alarm threshold, and the
topology is where it is recorded, next to the camera's IP. It changes only
when a unit is swapped, so it belongs in that reviewable file and the runtime
never asks a camera; this tool fills it in and re-checks it.

    python -m birdcount.deploy.tools.discover_ids            # report
    python -m birdcount.deploy.tools.discover_ids --verify   # compare with the topology (exit 1 on a mismatch)
    python -m birdcount.deploy.tools.discover_ids --write    # record the MACs in the topology

It also reports what each camera will get once mapped (threshold, mask): a
MAC with neither is a camera the alarm silently ignores.

MUST RUN ON THE FARM NETWORK: the camera subnet is firewalled from the office.
Credentials are CAMERA_USERNAME / CAMERA_PASSWORD, as for the video stream.
VAPIX is plain HTTP: `GET /axis-cgi/param.cgi?action=list&group=...` answers
`root.Network.eth0.MACAddress=B8:A4:4F:D5:1C:3C`.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from birdcount.core.camera_identity import index_by_mac, normalize_mac
from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.deploy.alarm import load_config
from birdcount.deploy.alarm.camera_ids import mac_of
from birdcount.deploy.settings import DeploySettings
from birdcount.deploy.topology import Camera, Topology, TopologyError, load_topology, set_camera_macs


# Newer firmware answers the first; the wildcard covers units whose interface is not eth0.
PARAM_GROUPS = ("Network.eth0.MACAddress", "Network.*.MACAddress")
_PARAM_LINE = re.compile(r"MACAddress\s*=\s*([0-9A-Fa-f:\-]{12,17})\s*$", re.MULTILINE)
# AXIS units ship self-signed certificates; this is a LAN-local identity query.
VERIFY_TLS = False


@dataclass
class Probe:
    zone: str
    camera: Camera
    mac: str | None = None
    error: str = ""


def parse_mac(text: str) -> str | None:
    """The MAC in a param.cgi answer, or None."""
    match = _PARAM_LINE.search(text)
    return (normalize_mac(match.group(1)) or None) if match else None


def fetch_mac(ip: str, auth: tuple[str, str], timeout: float) -> tuple[str | None, str]:
    """(mac, error) for one camera; never raises.

    Firmware differs in auth (digest or basic) and in which group has the MAC,
    so those are retried; a refused or timed-out connection is not — that just
    multiplies the wait for a camera that is off.
    """
    error = "no response"
    for scheme in ("https", "http"):
        mac, error = _probe_scheme(f"{scheme}://{ip}/axis-cgi/param.cgi", auth, timeout)
        if mac:
            return mac, ""
    return None, error


def _probe_scheme(url: str, auth: tuple[str, str], timeout: float) -> tuple[str | None, str]:
    import httpx

    error = "no response"
    for group in PARAM_GROUPS:
        for auth_obj in (httpx.DigestAuth(*auth), httpx.BasicAuth(*auth)):
            try:
                r = httpx.get(
                    url, params={"action": "list", "group": group}, auth=auth_obj, timeout=timeout, verify=VERIFY_TLS
                )
            except Exception as e:
                return None, f"{url.split(':')[0]}: {type(e).__name__}"  # unreachable: no point retrying
            if r.status_code == 401:
                error = "401 unauthorized (check CAMERA_USERNAME / CAMERA_PASSWORD)"
            elif r.status_code != 200:
                error = f"HTTP {r.status_code}"
            elif mac := parse_mac(r.text):
                return mac, ""
            else:
                error = "no MAC in param.cgi output"
    return None, error


def probe_all(topology: Topology, auth: tuple[str, str], timeout: float, workers: int) -> list[Probe]:
    probes = [Probe(zone.name, cam) for zone, cam in topology.cameras()]
    print(f"Querying {len(probes)} camera(s) over VAPIX (timeout {timeout:.0f} s)...", file=sys.stderr)

    def run(p: Probe) -> Probe:
        p.mac, p.error = fetch_mac(p.camera.ip, auth, timeout)
        return p

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, probes))


def report(probes: list[Probe], thresholds: dict[str, float], masks: dict) -> None:
    found = [p for p in probes if p.mac]
    print(f"\n{'zone':<22} {'IP':<16} {'MAC':<13} {'threshold':>9}  mask")
    for p in probes:
        if not p.mac:
            print(f"{p.zone[:22]:<22} {p.camera.ip:<16} {'-':<13} {'':>9}  unreachable: {p.error}")
            continue
        threshold, mask = thresholds.get(p.mac), masks.get(p.mac)
        flag = "" if threshold is not None and mask else "   <- incomplete"
        print(
            f"{p.zone[:22]:<22} {p.camera.ip:<16} {p.mac:<13} "
            f"{f'{threshold:.0f}' if threshold is not None else '-':>9}  {mask.name if mask else '-'}{flag}"
        )
    seen = {p.mac for p in found}
    if orphans := sorted(set(thresholds) - seen):
        print(f"\nThresholds with no camera on this network: {', '.join(orphans)}")
    if orphans := sorted(set(masks) - seen):
        print(f"Masks with no camera on this network: {', '.join(orphans)}")


def verify(probes: list[Probe]) -> int:
    wrong = [p for p in probes if p.mac and p.camera.mac and p.mac != p.camera.mac]
    unmapped = [p for p in probes if p.mac and not p.camera.mac]
    ok = sum(1 for p in probes if p.mac and p.mac == p.camera.mac)
    unreachable = sum(1 for p in probes if not p.mac)
    print(f"\nVerify: {ok} correct, {len(wrong)} WRONG, {len(unmapped)} unmapped, {unreachable} unreachable")
    for p in wrong:
        print(f"  WRONG  {p.zone} {p.camera.ip}: topology says {p.camera.mac}, the camera says {p.mac}")
    for p in unmapped:
        print(f"  unmapped  {p.zone} {p.camera.ip}: the camera says {p.mac}")
    if wrong:
        print("  Every WRONG camera is using another camera's mask and threshold right now.")
    return 1 if wrong else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--timeout", type=float, default=5.0, help="per request, seconds")
    parser.add_argument("--workers", type=int, default=8, help="cameras queried at once")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="compare with the topology; exit 1 on a mismatch")
    mode.add_argument("--write", action="store_true", help="record the discovered MACs in the topology")
    args = parser.parse_args(argv)

    try:
        import httpx  # noqa: F401
    except ImportError:
        print("This tool needs httpx: pip install -e .[deploy]", file=sys.stderr)
        return 2
    settings = DeploySettings.load()
    try:
        topology = load_topology(settings.topology_path)
    except TopologyError as e:
        print(e, file=sys.stderr)
        return 2
    auth = (settings.camera_username, settings.camera_password.get_secret_value())
    probes = probe_all(topology, auth, args.timeout, args.workers)
    if not any(p.mac for p in probes):
        print("\nNo camera answered. On the farm network? Credentials right?")
        return 2

    thresholds = {}
    if settings.alarm_config_path.is_file():
        thresholds = {mac_of(cid): c.threshold for cid, c in load_config(settings.alarm_config_path).cameras.items()}
    masks = {}
    if settings.mask_dir is not None and settings.mask_dir.is_dir():
        masks, _ = index_by_mac(p for p in settings.mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    report(probes, thresholds, masks)

    if args.verify:
        return verify(probes)
    if args.write:
        path = settings.topology_path
        text, missing = set_camera_macs(
            path.read_text(encoding="utf-8"), {p.camera.ip: p.mac for p in probes if p.mac}
        )
        path.write_text(text, encoding="utf-8")
        load_topology(path)  # the edit must leave a valid file
        print(f"\nRecorded {sum(1 for p in probes if p.mac) - len(missing)} MAC(s) in {path}.")
        if missing:
            print(f"Not found as a list entry of their own, edit by hand: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
