"""Manual control for the Tapo P100 smart plugs.

Reads credentials from the .env file (`TAPO_EMAIL` / `TAPO_PASSWORD`), the
same source the runtime uses — no credentials live in source. Device IPs are
pulled from `topology.yaml` by default, with an optional CLI override for
ad-hoc targets.

Usage:
    python -m tools.handlers.plug_switcher                # toggle every plug
    python -m tools.handlers.plug_switcher --action on    # force ON
    python -m tools.handlers.plug_switcher --action off   # force OFF
    python -m tools.handlers.plug_switcher --action status                  # query only
    python -m tools.handlers.plug_switcher --filter zone1,zone3 --timeout 5
    python -m tools.handlers.plug_switcher --target 192.168.0.185           # ad-hoc IP
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tapo import ApiClient


Action = str  # one of: "on" | "off" | "toggle" | "status"
_VALID_ACTIONS = ("on", "off", "toggle", "status")

_DEFAULT_TIMEOUT_SEC = 10.0


def load_credentials() -> tuple[str, str]:
    """Resolve TAPO_EMAIL / TAPO_PASSWORD from the environment / .env file."""
    load_dotenv()
    email = os.environ.get("TAPO_EMAIL", "").strip()
    password = os.environ.get("TAPO_PASSWORD", "").strip()
    if not email or not password:
        raise SystemExit(
            "TAPO_EMAIL and/or TAPO_PASSWORD are not set. Add them to your .env file (same vars the runtime uses)."
        )
    return email, password


def load_devices_from_topology() -> dict[str, str]:
    """Read `topology.yaml` and flatten to {label: ip} across all zones.

    Labels are `<zone-name>-<index>` so multiple plugs per zone don't collide.
    Returns an empty dict if topology.yaml is missing or has no plugs.
    """
    yaml_path = Path(__file__).resolve().parents[2] / "topology.yaml"
    if not yaml_path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    devices: dict[str, str] = {}
    for zone in data.get("zones", []):
        name = str(zone.get("name", "zone"))
        for i, ip in enumerate(zone.get("smart_plugs", []) or []):
            label = f"{name}-{i}" if len(zone.get("smart_plugs", [])) > 1 else name
            devices[label] = str(ip)
    return devices


async def _apply_action(client: ApiClient, label: str, ip: str, action: Action, timeout: float) -> str:
    """Run one device action; return a one-line status string for the summary."""
    try:
        device = await asyncio.wait_for(client.p100(ip), timeout=timeout)
        info = await asyncio.wait_for(device.get_device_info(), timeout=timeout)
        currently_on = bool(info.device_on)

        if action == "status":
            return f"[{label:<12}] {ip:<15}  state={'ON' if currently_on else 'OFF'}"

        if action == "on":
            want_on = True
        elif action == "off":
            want_on = False
        else:  # toggle
            want_on = not currently_on

        if want_on == currently_on:
            return f"[{label:<12}] {ip:<15}  already {'ON' if currently_on else 'OFF'} (no-op)"

        if want_on:
            await asyncio.wait_for(device.on(), timeout=timeout)
        else:
            await asyncio.wait_for(device.off(), timeout=timeout)
        return f"[{label:<12}] {ip:<15}  -> {'ON' if want_on else 'OFF'}"

    except TimeoutError:
        return f"[{label:<12}] {ip:<15}  TIMEOUT after {timeout:.1f}s"
    except Exception as exc:
        return f"[{label:<12}] {ip:<15}  ERROR: {type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual control for Tapo P100 smart plugs.")
    p.add_argument(
        "--action",
        choices=_VALID_ACTIONS,
        default="toggle",
        help="What to do on each device (default: toggle)",
    )
    p.add_argument(
        "--filter",
        default="",
        help="Comma-separated zone labels to act on (default: all). Ignored when --target is set.",
    )
    p.add_argument(
        "--target",
        default="",
        help="Override topology entirely — operate on this single IP (bypasses --filter).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SEC,
        help=f"Per-RPC timeout in seconds (default: {_DEFAULT_TIMEOUT_SEC})",
    )
    return p.parse_args()


def select_devices(args: argparse.Namespace) -> dict[str, str]:
    if args.target:
        return {"ad-hoc": args.target.strip()}

    devices = load_devices_from_topology()
    if not devices:
        raise SystemExit(
            "No plugs found. Either populate `topology.yaml` (zones[].smart_plugs) or pass --target <ip>."
        )

    if args.filter:
        wanted = {s.strip() for s in args.filter.split(",") if s.strip()}
        devices = {k: v for k, v in devices.items() if k in wanted}
        if not devices:
            raise SystemExit(f"--filter {args.filter!r} matched no zones in topology.yaml")

    return devices


async def main_async(args: argparse.Namespace) -> int:
    email, password = load_credentials()
    devices = select_devices(args)

    print(f"====== {args.action.upper()} on {len(devices)} device(s) ======")
    client = ApiClient(email, password)
    results = await asyncio.gather(
        *(_apply_action(client, label, ip, args.action, args.timeout) for label, ip in devices.items())
    )
    for line in results:
        print("  " + line)
    print(f"====== {sum(1 for r in results if 'ERROR' not in r and 'TIMEOUT' not in r)}/{len(results)} OK ======")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
