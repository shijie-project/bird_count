"""'Trigger All' button that flips every stream's hijack flag at once."""

import logging
import tkinter as tk
from collections.abc import Callable
from typing import Optional

from .._base import GuiComponent
from .._style import (
    BG_MONITOR_OFF,
    BG_MONITOR_OFF_HOVER,
    BG_PAGE,
    DOT_HIJACK,
    StatusSetter,
    status_at,
)
from .stream_grid import StreamGridComponent


logger = logging.getLogger(__name__)


class TriggerAllToggleButton(GuiComponent):
    """Toggles every hijack in `grid` to one common state."""

    def __init__(self, on_trigger_all: Callable[[bool], None], grid: StreamGridComponent):
        self.on_trigger_all = on_trigger_all
        self.grid = grid
        self.btn: Optional[tk.Button] = None
        self.set_status: Optional[StatusSetter] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=BG_PAGE)
        frame.pack(fill="x", padx=10, pady=(10, 0))
        self.btn = tk.Button(
            frame,
            text="TRIGGER ALL: ...",
            fg="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._on_click,
        )
        self.btn.pack(fill="x", pady=(0, 4))
        self._apply_appearance()

    def _on_click(self) -> None:
        target = not self.grid.any_hijacked()
        self.grid.set_all_hijacks(target)
        try:
            self.on_trigger_all(target)
        except Exception as e:
            logger.error("[TriggerAllButton] callback failed: %s", e, exc_info=True)
            return
        self._apply_appearance(target)
        if self.set_status:
            label = "ENABLED" if target else "DISABLED"
            self.set_status(
                status_at(f"All hijacks {label}"),
                fg=DOT_HIJACK if target else "#2980b9",
            )

    def refresh(self) -> None:
        self._apply_appearance()

    def _apply_appearance(self, state: Optional[bool] = None) -> None:
        if not self.btn:
            return
        if state is None:
            state = self.grid.any_hijacked()
        if state:
            self.btn.config(
                text="TRIGGER ALL: ON  (click to clear)",
                bg=DOT_HIJACK,
                activebackground="#e67e22",
            )
        else:
            self.btn.config(
                text="TRIGGER ALL: OFF  (click to enable all)",
                bg=BG_MONITOR_OFF,
                activebackground=BG_MONITOR_OFF_HOVER,
            )
