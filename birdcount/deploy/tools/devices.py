"""Switch zone speakers or smart plugs by hand, through the runtime's own drivers.

    python -m birdcount.deploy.tools.devices plug status
    python -m birdcount.deploy.tools.devices plug off --zone Zone_01_Right_Front
    python -m birdcount.deploy.tools.devices speaker on --ip 138.25.209.236
    python -m birdcount.deploy.tools.devices speaker toggle

Devices come from the topology (all of that kind, or `--zone`), or `--ip`
for any address. Credentials and the speaker clip come from the same
settings the runtime uses. `on` for a speaker plays the deterrent clip once.
Exit code 1 if any device failed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from birdcount.deploy.settings import DeploySettings
from birdcount.deploy.topology import TopologyError, load_topology


KINDS = {"speaker": "speakers", "plug": "smart_plugs"}
ACTIONS = ("on", "off", "status", "toggle")


def select(kind: str, zones: list[str], ips: list[str], settings: DeploySettings) -> list[tuple[str, str]]:
    """(label, ip) of every device to act on."""
    if ips:
        return [("--ip", ip) for ip in ips]
    topology = load_topology(settings.topology_path)
    unknown = set(zones) - {z.name for z in topology.zones}
    if unknown:
        raise SystemExit(f"No such zone(s) in {settings.topology_path.name}: {sorted(unknown)}")
    return [
        (zone.name, ip)
        for zone in topology.zones
        if not zones or zone.name in zones
        for ip in getattr(zone, KINDS[kind])
    ]


def driver_factory(kind: str, settings: DeploySettings):
    if kind == "speaker":
        from birdcount.deploy.handlers.speaker import SpeakerDriver

        auth = (settings.speaker_username, settings.speaker_password.get_secret_value())
        return lambda ip: SpeakerDriver(ip, auth, settings.speaker_audio_file)
    from birdcount.deploy.handlers.plug import PlugClient

    return PlugClient(settings.tapo_email, settings.tapo_password.get_secret_value())


async def apply(driver, action: str, timeout: float) -> str:
    """One device; returns its outcome line."""

    def state(on: bool | None) -> str:
        return "unknown" if on is None else "ON" if on else "OFF"

    async with driver:
        if action == "status":
            return state(await asyncio.wait_for(driver.is_on(), timeout))
        if action == "toggle":
            on = await asyncio.wait_for(driver.is_on(), timeout)
            if on is None:
                raise RuntimeError("state unknown, cannot toggle")
            action = "off" if on else "on"
        if action == "on":
            await asyncio.wait_for(driver.start(), timeout)
            await asyncio.wait_for(driver.refresh(), timeout)  # a speaker plays on refresh
        else:
            await asyncio.wait_for(driver.stop(), timeout)
        return f"-> {action.upper()}"


async def run(devices: list[tuple[str, str]], factory, action: str, timeout: float) -> int:
    async def one(label: str, ip: str) -> bool:
        try:
            print(f"  {label:<24} {ip:<16} {await apply(factory(ip), action, timeout)}")
            return True
        except Exception as e:
            print(f"  {label:<24} {ip:<16} FAILED: {type(e).__name__}: {e}")
            return False

    ok = await asyncio.gather(*(one(label, ip) for label, ip in devices))
    print(f"{sum(ok)}/{len(ok)} OK")
    return 0 if all(ok) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=sorted(KINDS))
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--zone", action="append", default=[], help="only this zone (repeatable)")
    parser.add_argument("--ip", action="append", default=[], help="this device instead of the topology's (repeatable)")
    parser.add_argument("--timeout", type=float, default=10.0, help="per request, seconds")
    args = parser.parse_args(argv)

    settings = DeploySettings.load()
    try:
        devices = select(args.kind, args.zone, args.ip, settings)
    except TopologyError as e:
        print(e, file=sys.stderr)
        return 2
    if not devices:
        print(f"No {KINDS[args.kind].replace('_', ' ')} selected.", file=sys.stderr)
        return 2
    try:
        factory = driver_factory(args.kind, settings)
    except ImportError as e:
        print(f"Missing driver library ({e}): pip install -e .[deploy]", file=sys.stderr)
        return 2
    print(f"{args.action.upper()} on {len(devices)} {args.kind}(s):")
    return asyncio.run(run(devices, factory, args.action, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
