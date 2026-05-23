"""Bicolor ON/OFF button bound to a MonitorTogglable handler."""

from .._base import MonitorTogglable
from ._bicolor_toggle import _BicolorToggleButton


class MonitorToggleButton(_BicolorToggleButton):
    """Click flips state via `handler.toggle()`; re-syncs each refresh so
    external state changes show up even without a click."""

    def __init__(self, handler: MonitorTogglable):
        super().__init__(
            label="MONITOR",
            toggle_fn=handler.toggle,
            is_enabled_fn=handler.is_enabled,
            log_tag="MonitorToggleButton",
            padx=12,
            pady=6,
        )
