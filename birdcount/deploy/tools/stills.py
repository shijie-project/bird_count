"""One still from every camera, now: what each one sees today.

    python -m birdcount.deploy.tools.stills                 # DATA_ROOT/stills/<YYYY-MM-DD_HHMM>/<MAC>.jpg
    python -m birdcount.deploy.tools.stills --zone Zone_01_Right_Front

For checking that a region mask still fits its camera (a unit bumped,
re-aimed or swapped since the mask was drawn, or since recordings that are
replayed): deploy.replay.alignment compares these with the recordings and the
masks. A still darker than `--dark` is reported: with the shed lights off it
shows nothing to compare.

AXIS VAPIX `GET /axis-cgi/jpg/image.cgi?resolution=1280x720`, with
CAMERA_USERNAME / CAMERA_PASSWORD (digest, else basic), on the farm network
or through the tunnel to it; a camera that does not answer is reported, not retried.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


RESOLUTION = "1280x720"  # the model's geometry (core.frames), so a still compares with counted frames
DARK = 25.0  # mean grey level (0-255) below which a still is the lights being off
VERIFY_TLS = False  # AXIS units ship self-signed certificates


def fetch_still(ip: str, auth: tuple[str, str], timeout: float) -> tuple[bytes | None, str]:
    """(JPEG, error) of one camera's still; never raises."""
    import httpx

    error = "no response"
    for scheme in ("https", "http"):
        url = f"{scheme}://{ip}/axis-cgi/jpg/image.cgi"
        for auth_obj in (httpx.DigestAuth(*auth), httpx.BasicAuth(*auth)):
            try:
                r = httpx.get(
                    url, params={"resolution": RESOLUTION}, auth=auth_obj, timeout=timeout, verify=VERIFY_TLS
                )
            except Exception as e:
                error = f"{scheme}: {type(e).__name__}"
                break  # unreachable over this scheme: the other auth will not help
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                return r.content, ""
            error = (
                "401 unauthorized (check CAMERA_USERNAME / CAMERA_PASSWORD)"
                if r.status_code == 401
                else f"HTTP {r.status_code}"
            )
    return None, error


def brightness(jpeg: bytes) -> float:
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return float(img.mean()) if img is not None else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m birdcount.deploy.tools.stills", description=__doc__.split("\n\n")[0])
    p.add_argument("--zone", default=None, help="only this zone's cameras")
    p.add_argument("--out", type=Path, default=None, help="default DATA_ROOT/stills/<YYYY-MM-DD_HHMM>/")
    p.add_argument("--dark", type=float, default=DARK, help="mean grey level below which a still is reported dark")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=4, help="cameras at once (the tunnel is slow)")
    args = p.parse_args(argv)

    from birdcount.deploy.settings import DeploySettings
    from birdcount.deploy.topology import TopologyError, load_topology

    settings = DeploySettings.load()
    try:
        topology = load_topology(settings.topology_path)
    except (OSError, TopologyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cams = [(z.name, c) for z, c in topology.cameras() if args.zone in (None, z.name)]
    if not cams:
        print("error: no cameras" + (f" in zone {args.zone}" if args.zone else ""), file=sys.stderr)
        return 2
    out = args.out or settings.data_root / "stills" / f"{datetime.now():%Y-%m-%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    auth = (settings.camera_username, settings.camera_password.get_secret_value())

    def grab(item):
        zone, cam = item
        jpeg, error = fetch_still(cam.ip, auth, args.timeout)
        return zone, cam, jpeg, error

    saved = dark = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for zone, cam, jpeg, error in pool.map(grab, cams):
            name = cam.mac or cam.ip.replace(".", "_")
            if jpeg is None:
                print(f"{zone[:22]:<22} {cam.ip:<16} {name:<13} unreachable: {error}")
                continue
            (out / f"{name}.jpg").write_bytes(jpeg)
            level = brightness(jpeg)
            saved += 1
            dark += level < args.dark
            print(
                f"{zone[:22]:<22} {cam.ip:<16} {name:<13} {level:5.0f}{'   DARK: lights off?' if level < args.dark else ''}"
            )
    print(f"\n{saved} of {len(cams)} still(s) in {out}" + (f"; {dark} dark" if dark else ""))
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
