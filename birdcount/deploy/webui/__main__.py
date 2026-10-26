"""The farm backend: every camera's state now, its day, its alarms, and labelling.

    python -m birdcount.deploy.webui                # http://127.0.0.1:8430, opens a browser tab
    python -m birdcount.deploy.webui --port 8500 --no-browser

A separate process from the runtime (`python -m birdcount.deploy`), reading
only the files the runtime writes: the live status file (STATUS_PATH), the
count log (COUNT_LOG_DIR) and the alarm evidence (ALARM_OUTPUT_DIR or the
alarm config's output_dir). It runs while the runtime is down and shows the
history; while the runtime runs it shows the live state within a second or two.
Same .env and configs as the runtime, so both look at the same farm.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m birdcount.deploy.webui", description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (this machine only by default)")
    p.add_argument("--port", type=int, default=8430)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from birdcount.core.log import setup_logging
    from birdcount.deploy.alarm import load_config
    from birdcount.deploy.settings import DeploySettings
    from birdcount.deploy.topology import TopologyError, load_topology
    from birdcount.deploy.webui.server import Backend, serve

    settings = DeploySettings.load()
    setup_logging(settings.log_level)
    try:
        config = load_config(settings.alarm_config_path)
        topology = load_topology(settings.topology_path)
    except (OSError, ValueError, KeyError, TopologyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    backend = Backend(
        config,
        topology,
        status_path=settings.status_path,
        count_log_dir=settings.count_log_dir,
        evidence_dir=settings.alarm_output_dir or config.output_dir,
        calibration_dir=settings.calibration_dir,
        calibration_model=settings.calibration_model,
    )
    url = f"http://{args.host}:{args.port}"

    def ready(_server) -> None:
        print(f"Farm backend: {url}  (Ctrl+C stops it)", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

    try:
        serve(backend, args.host, args.port, on_ready=ready)
    except OSError as e:
        print(f"error: cannot listen on {args.host}:{args.port}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
