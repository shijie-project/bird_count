"""A panel window: header, a vertical stack of widgets, a status bar."""

from __future__ import annotations

import contextlib
import logging
import time
import tkinter as tk

from birdcount.deploy.gui.widgets import HEADER, PAGE, Widget


logger = logging.getLogger(__name__)

REFRESH_HZ = 4.0


class Panel:
    def __init__(self, title: str, header: str, widgets: list[Widget], *, geometry: str | None = None):
        self.title, self.header, self.widgets, self.geometry = title, header, widgets, geometry
        self.root: tk.Misc | None = None
        self._status: tk.Label | None = None
        self._last_refresh = 0.0

    def open(self, master: tk.Misc | None = None) -> bool:
        """Build the window (a Toplevel of `master` when given). False without a display."""
        try:
            self.root = tk.Toplevel(master) if master is not None else tk.Tk()
        except tk.TclError as e:
            logger.error("No display for the %s panel: %s", self.header, e)
            return False
        self.root.title(self.title)
        self.root.configure(bg=PAGE)
        self.root.attributes("-topmost", True)
        if self.geometry:
            self.root.geometry(self.geometry)
            self.root.resizable(False, False)
        head = tk.Frame(self.root, bg=HEADER)
        head.pack(pady=10, fill="x")
        tk.Label(head, text=self.header, fg="white", bg=HEADER, font=("Segoe UI", 12, "bold")).pack()
        self._status = tk.Label(
            self.root, text="Ready...", bd=1, relief="sunken", anchor="w", font=("Segoe UI", 8), padx=10, pady=4
        )
        self._status.pack(fill="x", side="bottom")
        body = tk.Frame(self.root, bg=PAGE)
        body.pack(fill="both", expand=True)
        for w in self.widgets:
            try:
                w.mount(body, self.set_status)
            except Exception:
                logger.exception("Mounting %s failed.", type(w).__name__)
        return True

    def set_status(self, text: str, color: str | None = None) -> None:
        if self._status is not None:
            self._status.config(text=text, **({"fg": color} if color else {}))

    def refresh(self) -> None:
        """Refresh widgets at REFRESH_HZ; call often (the event loop is pumped separately)."""
        if self.root is None or time.monotonic() - self._last_refresh < 1.0 / REFRESH_HZ:
            return
        self._last_refresh = time.monotonic()
        for w in self.widgets:
            try:
                w.refresh()
            except Exception:
                logger.debug("Refreshing %s failed.", type(w).__name__, exc_info=True)

    def close(self) -> None:
        if self.root is not None:
            with contextlib.suppress(tk.TclError):
                self.root.destroy()
            self.root = None
