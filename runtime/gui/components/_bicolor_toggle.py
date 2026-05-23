"""Shared base for the ON/OFF toggle buttons (Monitor / DensityMap / Recorder).

Each button:
  - calls `toggle_fn()` on click,
  - reads current state via `is_enabled_fn()` to repaint on refresh,
  - shows ON/OFF colors + text built from `label`,
  - notifies the shell via `set_status` with a timestamped message.

Subclasses bind the right pair of callables in their `__init__` so the base
stays agnostic of which handler protocol they target — which avoids the
asymmetric method names (`toggle_density_map` vs `toggle`) leaking here.
"""

import logging
import tkinter as tk
from collections.abc import Callable

from .._base import GuiComponent
from .._style import (
    BG_MONITOR_OFF,
    BG_MONITOR_OFF_HOVER,
    BG_MONITOR_ON,
    BG_MONITOR_ON_HOVER,
    BG_PAGE,
    StatusSetter,
    status_at,
)


logger = logging.getLogger(__name__)


class _BicolorToggleButton(GuiComponent):
    """Two-state button with ON/OFF colors driven by external callables."""

    def __init__(
        self,
        *,
        label: str,
        toggle_fn: Callable[[], bool],
        is_enabled_fn: Callable[[], bool],
        log_tag: str,
        on_state_change: Callable[[bool], None] | None = None,
        pady: int | tuple[int, int] = 6,
        padx: int = 12,
    ):
        self._label = label
        self._toggle_fn = toggle_fn
        self._is_enabled_fn = is_enabled_fn
        self._log_tag = log_tag
        self._on_state_change = on_state_change
        self._pady = pady
        self._padx = padx

        self.btn: tk.Button | None = None
        self.set_status: StatusSetter | None = None

    def mount(self, parent: tk.Misc, set_status: StatusSetter | None = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=BG_PAGE)
        frame.pack(padx=self._padx, pady=self._pady, fill="x")
        self.btn = tk.Button(
            frame,
            text=f"{self._label}: ...",
            fg="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._on_click,
        )
        self.btn.pack(fill="x")
        self._apply_appearance()

    def refresh(self) -> None:
        self._apply_appearance()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_click(self) -> None:
        try:
            new_state = bool(self._toggle_fn())
        except Exception as e:
            logger.error("[%s] toggle failed: %s", self._log_tag, e, exc_info=True)
            return
        self._apply_appearance(new_state)
        if self._on_state_change is not None:
            try:
                self._on_state_change(new_state)
            except Exception as e:
                logger.error("[%s] on_state_change failed: %s", self._log_tag, e, exc_info=True)
        if self.set_status:
            label = "ON" if new_state else "OFF"
            self.set_status(
                status_at(f"{self._label} turned {label}"),
                fg=BG_MONITOR_ON if new_state else BG_MONITOR_OFF,
            )

    def _apply_appearance(self, state: bool | None = None) -> None:
        if not self.btn:
            return
        if state is None:
            try:
                state = bool(self._is_enabled_fn())
            except Exception:
                state = False
        if state:
            self.btn.config(
                text=f"{self._label}: ON  (click to turn OFF)",
                bg=BG_MONITOR_ON,
                activebackground=BG_MONITOR_ON_HOVER,
            )
        else:
            self.btn.config(
                text=f"{self._label}: OFF  (click to turn ON)",
                bg=BG_MONITOR_OFF,
                activebackground=BG_MONITOR_OFF_HOVER,
            )
