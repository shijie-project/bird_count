"""The training control panel.

    python -m birdcount.training.webui                  # http://127.0.0.1:8420, opens a browser tab
    python -m birdcount.training.webui --port 8500 --no-browser
    python -m birdcount.training.webui --host 0.0.0.0   # reachable from the LAN

Pages: Train (and sweeps), Evaluate, Video, Data (the dataset browser and the
file pipeline) and Label Studio (the server, its tunnel and the project
commands). Every run lands in OUTPUT_ROOT like a run started from a shell,
and runs started from a shell show up here too. Jobs and services the panel
started stop with it; running jobs are asked to stop as on Ctrl+C first.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser


_RESTARTED = "BIRDCOUNT_PANEL_RESTARTED"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m birdcount.training.webui", description="The training control panel.")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0: the whole LAN)")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from birdcount.core.log import setup_logging
    from birdcount.training.webui.server import Panel, serve
    from birdcount.training.webui.settings import WebuiSettings

    settings = WebuiSettings.load()
    setup_logging(settings.log_level)
    panel = Panel(settings)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    restarted = os.environ.pop(_RESTARTED, None) == "1"

    def ready(_server) -> None:
        print(f"Control panel: {url}  (Ctrl+C stops it)", flush=True)
        if not args.no_browser and not restarted:
            webbrowser.open(url)

    try:
        restart = serve(panel, args.host, args.port, on_ready=ready)
    except OSError as e:
        print(f"error: cannot listen on {args.host}:{args.port}: {e}", file=sys.stderr)
        return 1
    if restart:
        # A new process: imports, settings and code all start fresh.
        os.environ[_RESTARTED] = "1"
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable, "-m", "birdcount.training.webui", *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
