"""Scrollable per-stream grid: status dot + hijack button + optional REC button."""

import logging
import tkinter as tk
from collections.abc import Callable
from typing import Optional

from runtime.config import Config

from .._base import GuiComponent, RecorderTogglable
from .._style import (
    BG_MONITOR_ON,
    BG_MONITOR_ON_HOVER,
    BG_PAGE,
    DOT_ACTIVE,
    DOT_HIJACK,
    DOT_IDLE,
    StatusSetter,
    status_at,
)


logger = logging.getLogger(__name__)


class StreamGridComponent(GuiComponent):
    """The central widget of the debug panel.

    Owns per-stream hijack state and exposes a small API
    (`set_all_hijacks`, `any_hijacked`, `reset_hijacks`, `refresh_rec_buttons`)
    that sibling components (TriggerAll, RecorderToggle) call to coordinate
    without reaching into private state.
    """

    def __init__(
        self,
        config: Config,
        on_trigger: Callable[[int], None],
        recorder_handler: Optional[RecorderTogglable] = None,
        status_provider: Optional[Callable[[], dict]] = None,
        max_cols: int = 4,
    ):
        self.config = config
        self.on_trigger = on_trigger
        self.recorder_handler = recorder_handler
        self.status_provider = status_provider
        self.max_cols = max_cols

        self.set_status: Optional[StatusSetter] = None

        self.hijack_states: dict[int, bool] = dict.fromkeys(config.sid_to_ip, False)
        self.buttons: dict[int, tk.Button] = {}
        self.rec_buttons: dict[int, tk.Button] = {}
        self.dots: dict[int, tk.Label] = {}

        # Snapshot of the devices each stream owns — used to color the status
        # dot when any of those devices is currently active.
        self.stream_to_devices: dict[int, set[str]] = {}
        for sid, zone in config.sid_to_zone.items():
            ips: set[str] = set()
            ips.update(getattr(zone, "speakers", []) or [])
            ips.update(getattr(zone, "smart_plugs", []) or [])
            self.stream_to_devices[sid] = ips

    # ------------------------------------------------------------------
    # GuiComponent
    # ------------------------------------------------------------------

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.set_status = set_status

        container = tk.Frame(parent, bg=BG_PAGE)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(container, bg=BG_PAGE, highlightthickness=0, height=420)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        grid = tk.Frame(canvas, bg=BG_PAGE)
        canvas.create_window((0, 0), window=grid, anchor="nw")
        grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        for i, (stream_id, ip) in enumerate(self.config.sid_to_ip.items()):
            row, col = divmod(i, self.max_cols)
            self._build_cell(grid, row, col, stream_id, ip)

    def refresh(self) -> None:
        self._refresh_status_dots()
        for sid in self.rec_buttons:
            self._refresh_rec_button_style(sid)

    # ------------------------------------------------------------------
    # Public API for sibling components
    # ------------------------------------------------------------------

    def any_hijacked(self) -> bool:
        return any(self.hijack_states.values())

    def set_all_hijacks(self, state: bool) -> None:
        """Bulk-set every stream's hijack flag and re-render."""
        for sid in self.hijack_states:
            self.hijack_states[sid] = state
            self._refresh_button_style(sid)

    def reset_hijacks(self) -> None:
        """Clear all hijack flags. Called by the shell after an external Cancel All."""
        self.set_all_hijacks(False)

    def refresh_rec_buttons(self) -> None:
        """Re-pull the recorder state for every per-stream REC button."""
        for sid in self.rec_buttons:
            self._refresh_rec_button_style(sid)

    # ------------------------------------------------------------------
    # Cell construction
    # ------------------------------------------------------------------

    def _build_cell(self, parent: tk.Misc, row: int, col: int, stream_id: int, ip: str) -> None:
        cell = tk.Frame(parent, bg=BG_PAGE, padx=5, pady=5)
        cell.grid(row=row, column=col)

        dot = tk.Label(cell, text="●", fg=DOT_IDLE, bg=BG_PAGE, font=("Segoe UI", 10))
        dot.pack()
        self.dots[stream_id] = dot

        btn = tk.Button(
            cell,
            text=f"STREAM {stream_id}\n{ip}",
            width=16,
            height=3,
            bg="#ffffff",
            fg="#2c3e50",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cccccc",
            font=("Segoe UI", 9),
            command=lambda s=stream_id: self._handle_click(s),
        )
        btn.pack()
        self.buttons[stream_id] = btn

        btn.bind("<Enter>", lambda e, b=btn: b.config(highlightbackground="#3498db"))
        btn.bind("<Leave>", lambda e, b=btn, s=stream_id: self._refresh_button_style(s))

        if self.recorder_handler is not None:
            rec_btn = tk.Button(
                cell,
                text="REC",
                width=16,
                height=1,
                relief="flat",
                font=("Segoe UI", 9, "bold"),
                command=lambda s=stream_id: self._on_rec_click(s),
            )
            rec_btn.pack(pady=(2, 0))
            self.rec_buttons[stream_id] = rec_btn
            self._refresh_rec_button_style(stream_id)

    # ------------------------------------------------------------------
    # Hijack button
    # ------------------------------------------------------------------

    def _handle_click(self, stream_id: int) -> None:
        self.hijack_states[stream_id] = not self.hijack_states[stream_id]
        self._refresh_button_style(stream_id)
        try:
            self.on_trigger(stream_id)
        except Exception as e:
            logger.error("[StreamGrid] on_trigger callback failed for %s: %s", stream_id, e, exc_info=True)
        if self.set_status:
            state_text = "ENABLED" if self.hijack_states[stream_id] else "DISABLED"
            self.set_status(
                status_at(f"Stream {stream_id} Hijack {state_text}"),
                fg=DOT_HIJACK if self.hijack_states[stream_id] else "#2980b9",
            )

    def _refresh_button_style(self, stream_id: int) -> None:
        btn = self.buttons.get(stream_id)
        if not btn:
            return
        if self.hijack_states.get(stream_id, False):
            btn.config(bg=DOT_HIJACK, fg="white", highlightbackground="#e67e22")
        else:
            btn.config(bg="#ffffff", fg="#2c3e50", highlightbackground="#cccccc")

    # ------------------------------------------------------------------
    # REC button
    # ------------------------------------------------------------------

    def _on_rec_click(self, stream_id: int) -> None:
        if self.recorder_handler is None:
            return
        try:
            new_state = bool(self.recorder_handler.toggle_stream(stream_id))
        except Exception as e:
            logger.error("[StreamGrid] rec toggle failed for %s: %s", stream_id, e, exc_info=True)
            return
        self._refresh_rec_button_style(stream_id, new_state)
        if self.set_status:
            label = "ON" if new_state else "OFF"
            self.set_status(
                status_at(f"Stream {stream_id} Recording {label}"),
                fg=BG_MONITOR_ON if new_state else "#7f8c8d",
            )

    def _refresh_rec_button_style(self, stream_id: int, state: Optional[bool] = None) -> None:
        btn = self.rec_buttons.get(stream_id)
        if not btn or self.recorder_handler is None:
            return
        if state is None:
            try:
                state = bool(self.recorder_handler.is_stream_enabled(stream_id))
            except Exception:
                state = False
        if state:
            btn.config(text="REC ●", bg=BG_MONITOR_ON, fg="white", activebackground=BG_MONITOR_ON_HOVER)
        else:
            btn.config(text="REC", bg="#ffffff", fg="#2c3e50", activebackground="#dddddd")

    # ------------------------------------------------------------------
    # Status dots
    # ------------------------------------------------------------------

    def _refresh_status_dots(self) -> None:
        if not self.status_provider:
            return
        try:
            snapshot = self.status_provider() or {}
        except Exception as e:
            logger.debug("[StreamGrid] status_provider error: %s", e)
            return

        active_ips: set[str] = set()
        for ips in snapshot.values():
            if ips:
                active_ips.update(ips)

        for sid, dot in self.dots.items():
            if self.hijack_states.get(sid, False):
                color = DOT_HIJACK
            elif self.stream_to_devices.get(sid, set()) & active_ips:
                color = DOT_ACTIVE
            else:
                color = DOT_IDLE
            dot.config(fg=color)
