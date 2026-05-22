"""Bicolor ON/OFF button bound to a DensityMapTogglable handler."""

import logging
import time
import tkinter as tk
from typing import Optional

from .._base import DensityMapTogglable, GuiComponent
from .._style import (
    BG_MONITOR_OFF,
    BG_MONITOR_OFF_HOVER,
    BG_MONITOR_ON,
    BG_MONITOR_ON_HOVER,
    BG_PAGE,
    StatusSetter,
)


logger = logging.getLogger(__name__)


class DensityMapToggleButton(GuiComponent):
    """Click flips density-map overlay via `handler.toggle_density_map()`;
    re-syncs each refresh so external state changes show up even without a click."""

    def __init__(self, handler: DensityMapTogglable):
        self.handler = handler
        self.btn: Optional[tk.Button] = None
        self.set_status: Optional[StatusSetter] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=BG_PAGE)
        frame.pack(padx=10, pady=(0, 4), fill="x")
        self.btn = tk.Button(
            frame,
            text="DENSITY MAP: ...",
            fg="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._on_click,
        )
        self.btn.pack(fill="x")
        self._apply_appearance()

    def _on_click(self) -> None:
        try:
            new_state = bool(self.handler.toggle_density_map())
        except Exception as e:
            logger.error(f"[DensityMapToggleButton] toggle failed: {e}")
            return
        self._apply_appearance(new_state)
        if self.set_status:
            label = "ON" if new_state else "OFF"
            self.set_status(
                f"Density Map overlay turned {label} at {time.strftime('%H:%M:%S')}",
                fg=BG_MONITOR_ON if new_state else BG_MONITOR_OFF,
            )

    def _apply_appearance(self, state: Optional[bool] = None) -> None:
        if not self.btn:
            return
        if state is None:
            try:
                state = bool(self.handler.is_density_map_enabled())
            except Exception:
                state = False
        if state:
            self.btn.config(
                text="DENSITY MAP: ON  (click to turn OFF)",
                bg=BG_MONITOR_ON,
                activebackground=BG_MONITOR_ON_HOVER,
            )
        else:
            self.btn.config(
                text="DENSITY MAP: OFF  (click to turn ON)",
                bg=BG_MONITOR_OFF,
                activebackground=BG_MONITOR_OFF_HOVER,
            )

    def refresh(self) -> None:
        self._apply_appearance()
