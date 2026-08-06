"""Entrypoint for `python -m webui`.

Also survives being run as a plain script (`python webui/__main__.py`, or an
IDE's "run this file"), which otherwise fails on the relative import below.
"""

import argparse
import sys
import webbrowser
from pathlib import Path


if __package__:
    from .server import serve
else:  # started as a script: make the project root importable, then use the package
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from webui.server import serve


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the bird_count web UI")
    p.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 to expose on the LAN)")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--reload", action="store_true", help="auto-reload on source changes (development)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab on start")
    args = p.parse_args()

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"bird_count web UI -> {url}")
    if not args.no_browser and not args.reload:
        webbrowser.open(url)
    serve(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
