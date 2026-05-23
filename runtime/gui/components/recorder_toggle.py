"""Global ON/OFF button for the recorder handler; cascades visual state to the grid."""

import logging
import time
import tkinter as tk
from typing import Optional

from .._base import GuiComponent, RecorderTogglable
from .._style import (
    BG_MONITOR_OFF,
    BG_MONITOR_OFF_HOVER,
    BG_MONITOR_ON,
    BG_MONITOR_ON_HOVER,
    BG_PAGE,
    StatusSetter,
)
from .stream_grid import StreamGridComponent


logger = logging.getLogger(__name__)


class RecorderToggleButton(GuiComponent):
    """Click flips recorder.toggle() then refreshes every per-stream REC button."""

    def __init__(self, handler: RecorderTogglable, grid: StreamGridComponent):
        self.handler = handler
        self.grid = grid
        self.btn: Optional[tk.Button] = None
        self.set_status: Optional[StatusSetter] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=BG_PAGE)
        frame.pack(fill="x", padx=10)
        self.btn = tk.Button(
            frame,
            text="RECORDING: ...",
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
            new_state = bool(self.handler.toggle())
        except Exception as e:
            logger.error(f"[RecorderToggleButton] toggle failed: {e}")
            return
        self._apply_appearance(new_state)
        self.grid.refresh_rec_buttons()
        if self.set_status:
            label = "ON" if new_state else "OFF"
            self.set_status(
                f"Recording turned {label} at {time.strftime('%H:%M:%S')}",
                fg=BG_MONITOR_ON if new_state else "#7f8c8d",
            )

    def refresh(self) -> None:
        self._apply_appearance()

    def _apply_appearance(self, state: Optional[bool] = None) -> None:
        if not self.btn:
            return
        if state is None:
            try:
                state = bool(self.handler.is_enabled())
            except Exception:
                state = False
        if state:
            self.btn.config(
                text="RECORDING: ON  (click to turn OFF)",
                bg=BG_MONITOR_ON,
                activebackground=BG_MONITOR_ON_HOVER,
            )
        else:
            self.btn.config(
                text="RECORDING: OFF  (click to turn ON)",
                bg=BG_MONITOR_OFF,
                activebackground=BG_MONITOR_OFF_HOVER,
            )
