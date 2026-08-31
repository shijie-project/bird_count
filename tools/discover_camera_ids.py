"""Ask every AXIS camera in topology.yaml for its MAC, over VAPIX.

An IP says nothing about which camera it is, but two per-camera assets are keyed
on the camera's identity rather than its address: the region mask
(`MASK_DIR/<MAC>.png`) and the pile-up threshold (`configs/alarm.json`). The
link between them is `camera_ids:` in topology.yaml — a bare MAC per camera,
positionally aligned with `cameras:`.

That mapping is deployment fact, not runtime state: it changes only when
somebody re-IPs or swaps a unit. So it belongs in a reviewable file, and the
runtime never queries a camera. This tool fills the file in, and can re-check it
later.

    python -m tools.discover_camera_ids                 # print a paste-ready block
    python -m tools.discover_camera_ids --verify        # compare against topology.yaml
    python -m tools.discover_camera_ids --write         # edit topology.yaml in place

Beyond the MAC, it reports what each camera would actually get once mapped —
a threshold, a mask, both or neither — because a MAC that resolves to no
threshold and no mask is a camera the alarm silently ignores.

MUST RUN ON THE FARM NETWORK. The camera subnet is firewalled from the office
VLAN, so from a desk this reports every camera as unreachable.

VAPIX is just HTTP: `GET /axis-cgi/param.cgi?action=list&group=<group>` returns
`root.Network.eth0.MACAddress=B8:A4:4F:D5:1C:3C` as plain text.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


try:
    import httpx
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    httpx = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.camera_identity import index_by_mac, normalize_mac  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = ROOT / "topology.yaml"

# Newer firmware answers the first; the wildcard covers units whose interface
# is not named eth0 (some AXIS models expose `eth0` only over the second form).
_PARAM_GROUPS = ("Network.eth0.MACAddress", "Network.*.MACAddress")

# `root.Network.eth0.MACAddress=B8:A4:4F:D5:1C:3C`
_PARAM_LINE = re.compile(r"MACAddress\s*=\s*([0-9A-Fa-f:\-]{12,17})\s*$", re.MULTILINE)

# AXIS units ship self-signed certificates, so TLS verification is off by
# design here — same as the MJPEG pull in `runtime.config._camera_url`. This is
# a LAN-local identity query, not a trust boundary.
_VERIFY_TLS = False


@dataclass
class Probe:
    """One camera's answer (or lack of one)."""

    ip: str
    zone: str
    slot: int  # index within the zone's `cameras:` list
    mac: Optional[str] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.mac)


def _probe_scheme(ip: str, scheme: str, auth: tuple[str, str], timeout: float) -> tuple[Optional[str], str]:
    """Try one scheme against one camera. A transport failure ends it immediately.

    Firmware differs in both auth (digest on current releases, basic on older)
    and in which param group exposes the MAC, so those are worth retrying. A
    refused or timed-out connection is not — retrying it just multiplies the
    wait for a camera that is simply off.
    """
    url = f"{scheme}://{ip}/axis-cgi/param.cgi"
    last_error = "no response"
    for group in _PARAM_GROUPS:
        for auth_obj in (httpx.DigestAuth(*auth), httpx.BasicAuth(*auth)):
            try:
                response = httpx.get(
                    url,
                    params={"action": "list", "group": group},
                    auth=auth_obj,
                    timeout=timeout,
                    verify=_VERIFY_TLS,
                )
            except Exception as e:
                return None, f"{scheme}: {type(e).__name__}"
            if response.status_code == 401:
                last_error = "401 unauthorized (wrong --username/--password?)"
                continue
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                continue
            match = _PARAM_LINE.search(response.text)
            if match and (mac := normalize_mac(match.group(1))):
                return mac, ""
            last_error = "MAC not present in param.cgi output"
    return None, last_error


def _fetch_mac(ip: str, auth: tuple[str, str], timeout: float) -> tuple[Optional[str], str]:
    """Return `(mac, error)` for one camera. Never raises."""
    if httpx is None:
        return None, "httpx not installed (pip install -r requirements.txt)"

    last_error = "no response"
    for scheme in ("https", "http"):
        mac, last_error = _probe_scheme(ip, scheme, auth, timeout)
        if mac:
            return mac, ""
    return None, last_error


def _load_zones() -> list[dict]:
    if not TOPOLOGY.exists():
        sys.exit(f"topology.yaml not found at {TOPOLOGY}")
    data = yaml.safe_load(TOPOLOGY.read_text(encoding="utf-8")) or {}
    zones = data.get("zones") or []
    if not zones:
        sys.exit(f"No zones defined in {TOPOLOGY}")
    return zones


def _probe_all(zones: list[dict], auth: tuple[str, str], timeout: float, workers: int) -> list[Probe]:
    probes = [
        Probe(ip=str(ip), zone=zone.get("name", f"zone{zi}"), slot=si)
        for zi, zone in enumerate(zones)
        for si, ip in enumerate(zone.get("cameras") or [])
    ]
    if not probes:
        sys.exit("No cameras listed in topology.yaml")

    print(f"Querying {len(probes)} camera(s) over VAPIX (timeout {timeout:.0f}s)...\n", file=sys.stderr)

    def run(probe: Probe) -> Probe:
        probe.mac, probe.error = _fetch_mac(probe.ip, auth, timeout)
        return probe

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, probes))


def _known_thresholds(alarm_config: Path) -> dict[str, float]:
    """MAC -> pile-up threshold, from the alarm config's `axisN/MAC` keys."""
    if not alarm_config.exists():
        return {}
    try:
        cameras = json.loads(alarm_config.read_text(encoding="utf-8")).get("cameras", [])
    except Exception as e:
        print(f"[warn] could not read {alarm_config}: {e}", file=sys.stderr)
        return {}
    return {str(c["camera_id"]).rsplit("/", 1)[-1].upper(): c.get("threshold") for c in cameras}


def _known_masks(mask_dir: Optional[Path]) -> dict[str, Path]:
    if not mask_dir or not mask_dir.is_dir():
        return {}
    files = [p for p in sorted(mask_dir.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    unique, _ = index_by_mac(files)
    return unique


def _report_coverage(probes: list[Probe], thresholds: dict, masks: dict) -> None:
    """Say what each discovered camera would actually get once mapped."""
    found = [p for p in probes if p.ok]
    if not found:
        return

    print("\nCoverage once mapped:")
    print(f"  {'IP':<16} {'MAC':<14} {'threshold':>9}  mask")
    print("  " + "-" * 52)
    for probe in found:
        thr = thresholds.get(probe.mac)
        mask = masks.get(probe.mac)
        thr_text = f"{thr:.0f}" if thr is not None else "--"
        mask_text = mask.name if mask else "--"
        flag = "" if (thr is not None and mask) else "   <- incomplete"
        print(f"  {probe.ip:<16} {probe.mac:<14} {thr_text:>9}  {mask_text}{flag}")

    orphan_thr = set(thresholds) - {p.mac for p in found}
    orphan_mask = set(masks) - {p.mac for p in found}
    if orphan_thr:
        print(f"\n  Thresholds with no camera on this network: {', '.join(sorted(orphan_thr))}")
    if orphan_mask:
        print(f"  Masks with no camera on this network:      {', '.join(sorted(orphan_mask))}")


def _print_block(zones: list[dict], probes: list[Probe]) -> None:
    """Emit the `camera_ids:` lists, ready to paste into topology.yaml."""
    by_zone: dict[str, dict[int, Probe]] = {}
    for probe in probes:
        by_zone.setdefault(probe.zone, {})[probe.slot] = probe

    print("\n" + "=" * 60)
    print("Paste each block under its zone, alongside `cameras:`")
    print("=" * 60)
    for zi, zone in enumerate(zones):
        name = zone.get("name", f"zone{zi}")
        cameras = zone.get("cameras") or []
        if not cameras:
            continue
        print(f"\n  - name: {name!r}")
        print("    camera_ids:")
        for si, ip in enumerate(cameras):
            probe = by_zone.get(name, {}).get(si)
            if probe and probe.ok:
                print(f'      - "{probe.mac}"    # {ip}')
            else:
                reason = probe.error if probe else "not probed"
                # An empty entry keeps the positional alignment while leaving
                # this camera honestly uncovered.
                print(f'      - ""              # {ip} - UNRESOLVED: {reason}')


def _verify(zones: list[dict], probes: list[Probe]) -> int:
    """Compare live MACs against topology.yaml. Returns a process exit code."""
    by_zone: dict[str, dict[int, Probe]] = {}
    for probe in probes:
        by_zone.setdefault(probe.zone, {})[probe.slot] = probe

    mismatches, missing, unreachable, ok = [], [], [], 0
    for zi, zone in enumerate(zones):
        name = zone.get("name", f"zone{zi}")
        cameras = zone.get("cameras") or []
        configured = list(zone.get("camera_ids") or [])
        for si, ip in enumerate(cameras):
            probe = by_zone.get(name, {}).get(si)
            want = normalize_mac(str(configured[si])) if si < len(configured) else ""
            if probe is None or not probe.ok:
                unreachable.append((name, ip, probe.error if probe else "not probed"))
            elif not want:
                missing.append((name, ip, probe.mac))
            elif want != probe.mac:
                mismatches.append((name, ip, want, probe.mac))
            else:
                ok += 1

    print(f"\nVerify: {ok} correct, {len(mismatches)} WRONG, {len(missing)} unmapped, {len(unreachable)} unreachable")

    if mismatches:
        print("\n  WRONG — topology.yaml disagrees with the camera. Every one of these")
        print("  is applying another camera's threshold and mask right now:")
        for zone_name, ip, want, got in mismatches:
            print(f"    {zone_name} {ip}: topology says {want}, camera says {got}")
    if missing:
        print("\n  Unmapped — no camera_ids entry, so no mask and no alarm coverage:")
        for zone_name, ip, got in missing:
            print(f"    {zone_name} {ip}: camera says {got}")
    if unreachable:
        print("\n  Unreachable — not checked (are you on the farm network?):")
        for zone_name, ip, err in unreachable:
            print(f"    {zone_name} {ip}: {err}")

    return 1 if mismatches else 0


def _write_back(zones: list[dict], probes: list[Probe]) -> None:
    """Rewrite topology.yaml's `camera_ids:` from the discovered MACs.

    Rewrites via the YAML round-trip, which drops the file's comments — so the
    original is kept next to it as `.bak` rather than being overwritten blind.
    """
    by_zone: dict[str, dict[int, Probe]] = {}
    for probe in probes:
        by_zone.setdefault(probe.zone, {})[probe.slot] = probe

    raw = yaml.safe_load(TOPOLOGY.read_text(encoding="utf-8"))
    for zi, zone in enumerate(raw.get("zones") or []):
        name = zone.get("name", f"zone{zi}")
        cameras = zone.get("cameras") or []
        if not cameras:
            continue
        resolved = [by_zone.get(name, {}).get(si) for si in range(len(cameras))]
        zone["camera_ids"] = [p.mac if (p and p.ok) else "" for p in resolved]

    backup = TOPOLOGY.with_suffix(".yaml.bak")
    backup.write_text(TOPOLOGY.read_text(encoding="utf-8"), encoding="utf-8")
    TOPOLOGY.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nWrote {TOPOLOGY} (comments dropped by the YAML round-trip; original saved to {backup.name}).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover AXIS camera MACs over VAPIX and map them to topology.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Must run on the farm network — the camera subnet is firewalled from the office VLAN.",
    )
    parser.add_argument("--username", default="root", help="VAPIX user (default: root, as used for the MJPEG pull)")
    parser.add_argument("--password", default="root", help="VAPIX password (default: root)")
    parser.add_argument("--timeout", type=float, default=5.0, help="per-request timeout, seconds (default: 5)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent probes (default: 8)")
    parser.add_argument("--alarm-config", type=Path, default=ROOT / "configs" / "alarm.json")
    parser.add_argument("--mask-dir", type=Path, default=None, help="cross-check masks (default: MASK_DIR from .env)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="compare live MACs against topology.yaml, then exit")
    mode.add_argument("--write", action="store_true", help="rewrite topology.yaml's camera_ids in place")
    args = parser.parse_args()

    if httpx is None:
        sys.exit("httpx is required: pip install -r requirements.txt")

    mask_dir = args.mask_dir
    if mask_dir is None:
        import dotenv

        dotenv.load_dotenv(ROOT / ".env")
        raw = os.getenv("MASK_DIR", "").strip()
        mask_dir = (ROOT / raw).resolve() if raw else None

    zones = _load_zones()
    probes = _probe_all(zones, (args.username, args.password), args.timeout, args.workers)

    found = sum(p.ok for p in probes)
    print(f"Reached {found}/{len(probes)} camera(s).")
    for probe in probes:
        if not probe.ok:
            print(f"  unreachable: {probe.ip:<16} ({probe.zone}) - {probe.error}")
    if not found:
        # stdout, so it lands after the per-camera list rather than ahead of it.
        print("\nNothing answered. Are you on the farm network? Check --username/--password.")
        return 2

    _report_coverage(probes, _known_thresholds(args.alarm_config), _known_masks(mask_dir))

    if args.verify:
        return _verify(zones, probes)
    if args.write:
        _write_back(zones, probes)
        return 0
    _print_block(zones, probes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
