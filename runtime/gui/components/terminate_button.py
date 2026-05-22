"""Last-resort 'Terminate Program' button with a confirmation dialog."""

import logging
import time
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Optional

from .._base import GuiComponent
from .._style import BG_CANCEL, BG_CANCEL_HOVER, BG_PAGE, StatusSetter


logger = logging.getLogger(__name__)


class TerminateButton(GuiComponent):
    def __init__(self, on_terminate: Callable[[], None]):
        self.on_terminate = on_terminate
        self.btn: Optional[tk.Button] = None
        self.set_status: Optional[StatusSetter] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=BG_PAGE)
        frame.pack(fill="x", padx=10, pady=(8, 4))
        self.btn = tk.Button(
            frame,
            text="TERMINATE PROGRAM",
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
            title="Confirm Terminate",
            message=(
                "Shut down the entire bird-count program?\n\n"
                "This stops capture, inference, recording, alerts, and the GUI. "
                "Running recordings will be finalized cleanly."
            ),
            parent=self.btn.winfo_toplevel(),
            icon="warning",
        )
        if not confirmed:
            return
        try:
            self.on_terminate()
        except Exception as e:
            logger.error(f"[TerminateButton] callback failed: {e}")
            return
        if self.set_status:
            self.set_status(
                f"Termination requested at {time.strftime('%H:%M:%S')}; shutting down...",
                fg=BG_CANCEL,
            )
        # Disable so the user can't double-click while shutdown is in flight.
        self.btn.config(state="disabled", text="TERMINATING...")
