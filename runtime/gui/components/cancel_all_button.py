"""Red 'Cancel All Alerts' button with confirmation dialog."""

import logging
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox

from .._base import GuiComponent
from .._style import BG_CANCEL, BG_CANCEL_HOVER, BG_PAGE, StatusSetter, status_at


logger = logging.getLogger(__name__)


class CancelAllButton(GuiComponent):
    def __init__(self, on_cancel_all: Callable[[], None]):
        self.on_cancel_all = on_cancel_all
        self.btn: tk.Button | None = None
        self.set_status: StatusSetter | None = None

    def mount(self, parent: tk.Misc, set_status: StatusSetter | None = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=BG_PAGE)
        frame.pack(padx=12, pady=(12, 6), fill="x")
        self.btn = tk.Button(
            frame,
            text="CANCEL ALL ALERTS",
            bg=BG_CANCEL,
            fg="white",
            activebackground=BG_CANCEL_HOVER,
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._on_click,
        )
        self.btn.pack(fill="x")
        self.btn.bind("<Enter>", lambda e: self.btn.config(bg=BG_CANCEL_HOVER))
        self.btn.bind("<Leave>", lambda e: self.btn.config(bg=BG_CANCEL))

    def _on_click(self) -> None:
        if not self.btn:
            return
        confirmed = messagebox.askyesno(
            title="Confirm Cancel All",
            message=(
                "Cancel all currently active alerts?\n\n"
                "This will turn off every triggered device that was automatically "
                "switched on after a bird-count threshold was exceeded."
            ),
            parent=self.btn.winfo_toplevel(),
        )
        if not confirmed:
            return
        try:
            self.on_cancel_all()
        except Exception as e:
            logger.error("[CancelAllButton] callback failed: %s", e, exc_info=True)
        if self.set_status:
            self.set_status(status_at("All alerts cancelled"), fg=BG_CANCEL)
