"""Read-only label showing total + per-handler active device counts."""

import tkinter as tk
from typing import Callable, Optional

from .._base import GuiComponent
from .._style import BG_PAGE, StatusSetter


class ActivityLabel(GuiComponent):
    def __init__(self, status_provider: Callable[[], dict]):
        self.status_provider = status_provider
        self.label: Optional[tk.Label] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.label = tk.Label(
            parent,
            text="Active devices: 0",
            bg=BG_PAGE,
            fg="#2c3e50",
            font=("Segoe UI", 9),
            pady=4,
        )
        self.label.pack(fill="x")

    def refresh(self) -> None:
        if not self.label:
            return
        try:
            snapshot = self.status_provider() or {}
        except Exception:
            return
        total = sum(len(v) for v in snapshot.values() if v)
        parts = [f"{k}={len(v)}" for k, v in snapshot.items() if v]
        suffix = f"  ({', '.join(parts)})" if parts else ""
        try:
            self.label.config(text=f"Active devices: {total}{suffix}")
        except tk.TclError:
            pass
