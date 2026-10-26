"""The HTTP plumbing both local web apps share: routing, guards, JSON, files.

Python's own threading HTTP server is enough for one person's local page and
keeps the process small. `LocalHandler` adds what each app would otherwise
repeat:

* the Host header must name this machine, so a web page cannot reach the app
  through DNS rebinding;
* anything but GET must come from the app's own page (Origin), so another
  site open in the browser cannot change anything, and must be JSON;
* routes are plain functions `fn(handler, query, *groups)` returning a JSON
  value (or None when they sent the response themselves);
* files go out with Last-Modified revalidation and byte ranges (video seeking).

stdlib only.
"""

from __future__ import annotations

import email.utils
import ipaddress
import json
import logging
import mimetypes
import re
import socket
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit


logger = logging.getLogger(__name__)

MAX_JSON_BODY = 2 * 2**20
CONTENT_TYPES = {  # the Windows registry can map these to text/plain, which breaks modules
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".jsonl": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".mkv": "video/x-matroska",
}

Route = Callable[..., object]


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _hostname(host: str) -> str:
    if host.startswith("["):
        return host[1 : host.find("]")].lower()
    return host.rsplit(":", 1)[0].lower()


def _local_host(name: str, bind: str) -> bool:
    if name in ("localhost", bind.lower(), socket.gethostname().lower()):
        return True
    try:
        ipaddress.ip_address(name)  # an address, not a name a page's DNS could re-point
        return True
    except ValueError:
        return False


class LocalHandler(BaseHTTPRequestHandler):
    """Subclasses set `routes` {(method, path): fn} and `pattern_routes` [((method, regex), fn)]."""

    protocol_version = "HTTP/1.1"
    server_version = "birdcount"
    routes: ClassVar[dict[tuple[str, str], Route]] = {}
    pattern_routes: ClassVar[list[tuple[tuple[str, re.Pattern], Route]]] = []

    def log_message(self, fmt: str, *args) -> None:  # the default logs every poll
        pass

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _route(self, method: str, path: str) -> Route | None:
        route = self.routes.get((method, path))
        if route is not None:
            return route
        for (m, pattern), fn in self.pattern_routes:
            if m == method and (match := pattern.fullmatch(path)):
                return lambda h, q, fn=fn, match=match: fn(h, q, *match.groups())
        return None

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            self._guard(method)
            route = self._route(method, url.path)
            if route is None:
                raise HttpError(404, f"no route {method} {url.path}")
            result = route(self, query)
            if result is not None:
                self.send_json(result)
        except HttpError as e:
            self.send_json({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            logger.exception("%s %s failed", method, self.path)
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _guard(self, method: str) -> None:
        host = self.headers.get("Host", "")
        if not _local_host(_hostname(host), self.server.server_address[0]):
            raise HttpError(403, "unexpected Host header")
        if method != "GET":
            origin = self.headers.get("Origin")
            if (origin and origin != f"http://{host}") or self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise HttpError(403, "requests must come from this app's own page")

    # --- responses -----------------------------------------------------------------------------

    def send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, separators=(",", ":"), default=str).encode()
        self.send(status, body, "application/json", {"Cache-Control": "no-store"})

    def json_body(self) -> dict:
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise HttpError(415, "expected application/json")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_JSON_BODY:
            raise HttpError(413, "request too large")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise HttpError(400, "invalid JSON") from None
        if not isinstance(data, dict):
            raise HttpError(400, "expected a JSON object")
        return data

    def send_file(self, path: Path, *, cache: str = "no-cache") -> None:
        """A file, with Last-Modified revalidation and byte ranges (video seeking)."""
        st = path.stat()
        last_modified = email.utils.formatdate(st.st_mtime, usegmt=True)
        if self.headers.get("If-Modified-Since") == last_modified:
            self.send_response(304)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        suffix = path.suffix.lower()
        content_type = CONTENT_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size, start, end = st.st_size, 0, st.st_size - 1
        status = 200
        if (rng := re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))) and size:
            if rng[1]:
                start, end = int(rng[1]), min(int(rng[2]) if rng[2] else size - 1, size - 1)
            elif rng[2]:  # the last N bytes
                start = max(0, size - int(rng[2]))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_static(self, folder: Path, name: str) -> None:
        """A file of the app's own page, never one outside `folder`."""
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise HttpError(404, "no such file")
        self.send_file(path)
