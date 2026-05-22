"""Shared visual style + small type aliases used across the GUI package."""

from typing import Callable


BG_PAGE = "#f0f0f0"
BG_HEADER = "#2c3e50"
BG_CANCEL = "#c0392b"
BG_CANCEL_HOVER = "#e74c3c"
BG_MONITOR_ON = "#27ae60"
BG_MONITOR_ON_HOVER = "#2ecc71"
BG_MONITOR_OFF = "#7f8c8d"
BG_MONITOR_OFF_HOVER = "#95a5a6"
DOT_IDLE = "#bdc3c7"
DOT_HIJACK = "#f39c12"
DOT_ACTIVE = "#e74c3c"

# fg is the only well-known kwarg passed through; signature stays loose so
# components can add their own Tk options later without renegotiating the API.
StatusSetter = Callable[..., None]
