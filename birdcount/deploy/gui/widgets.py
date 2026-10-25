"""Panel widgets. Each mounts itself into a parent frame and may refresh periodically."""

from __future__ import annotations

import logging
import time
import tkinter as tk
from collections.abc import Callable, Mapping

from birdcount.deploy.gui.controls import ControlState, Toggle


logger = logging.getLogger(__name__)

PAGE = "#f0f0f0"
HEADER = "#2c3e50"
RED, RED_HOVER = "#c0392b", "#e74c3c"
ON, ON_HOVER = "#27ae60", "#2ecc71"
OFF, OFF_HOVER = "#7f8c8d", "#95a5a6"
HIJACK, HIJACK_HOVER = "#f39c12", "#e67e22"
DOT_IDLE, DOT_ACTIVE, DOT_OFFLINE = "#bdc3c7", "#e74c3c", "#34495e"
_BOLD = ("Segoe UI", 11, "bold")

StatusSetter = Callable[[str, str | None], None]


def stamped(message: str) -> str:
    return f"{message} at {time.strftime('%H:%M:%S')}"


class Widget:
    def mount(self, parent: tk.Misc, status: StatusSetter) -> None:
        raise NotImplementedError

    def refresh(self) -> None:
        """Called a few times a second; re-read whatever the widget shows."""


class ConfirmButton(Widget):
    """A big red button that acts on a second click within `CONFIRM_SECONDS`
    (Cancel All, Terminate).

    Not a modal dialog, as in v2: a dialog runs an event loop of its own, so
    the consumer loop that pumps the panels — and processes every batch —
    stood still for as long as the dialog was open: no counts, no alarms.
    """

    CONFIRM_SECONDS = 5.0

    def __init__(self, text: str, warning: str, on_confirm: Callable[[], None], done: str, *, once: bool = False):
        self.text, self.warning, self.on_confirm, self.done, self.once = text, warning, on_confirm, done, once
        self.button: tk.Button | None = None
        self._status: StatusSetter | None = None
        self._armed_until = 0.0

    def mount(self, parent: tk.Misc, status: StatusSetter) -> None:
        self._status = status
        frame = tk.Frame(parent, bg=PAGE)
        frame.pack(fill="x", padx=12, pady=(10, 4))
        self.button = tk.Button(
            frame,
            text=self.text,
            bg=RED,
            fg="white",
            activebackground=RED_HOVER,
            activeforeground="white",
            relief="flat",
            font=_BOLD,
            height=2,
            command=self.click,
        )
        self.button.pack(fill="x")

    @property
    def armed(self) -> bool:
        return time.monotonic() < self._armed_until

    def click(self) -> None:
        if not self.armed:
            self._armed_until = time.monotonic() + self.CONFIRM_SECONDS
            self.button.config(text=f"CLICK AGAIN TO CONFIRM ({self.CONFIRM_SECONDS:.0f} s)", bg=HIJACK)
            self._status(self.warning, RED)
            return
        self._disarm()
        try:
            self.on_confirm()
        except Exception:
            logger.exception("%s failed.", self.text)
            return
        self._status(stamped(self.done), RED)
        if self.once:
            self.button.config(state="disabled")

    def refresh(self) -> None:
        if self._armed_until and not self.armed:
            self._disarm()

    def _disarm(self) -> None:
        self._armed_until = 0.0
        self.button.config(text=self.text, bg=RED)


class ToggleButton(Widget):
    """Green ON / grey OFF; re-reads its state on refresh, so outside changes show."""

    def __init__(self, toggle: Toggle):
        self.toggle = toggle
        self.button: tk.Button | None = None
        self._status: StatusSetter | None = None

    def mount(self, parent: tk.Misc, status: StatusSetter) -> None:
        self._status = status
        frame = tk.Frame(parent, bg=PAGE)
        frame.pack(fill="x", padx=10, pady=(4, 4))
        self.button = tk.Button(frame, fg="white", relief="flat", font=_BOLD, height=2, command=self.click)
        self.button.pack(fill="x")
        self.refresh()

    def click(self) -> None:
        on = not self.toggle.get()
        try:
            self.toggle.set(on)
        except Exception:
            logger.exception("Switching %s failed.", self.toggle.label)
            return
        self.refresh()
        self._status(stamped(f"{self.toggle.label} turned {'ON' if on else 'OFF'}"), ON if on else OFF)

    def refresh(self) -> None:
        on = self.toggle.get()
        self.button.config(
            text=f"{self.toggle.label}: {'ON' if on else 'OFF'}  (click to turn {'OFF' if on else 'ON'})",
            bg=ON if on else OFF,
            activebackground=ON_HOVER if on else OFF_HOVER,
        )


class ActivityLabel(Widget):
    """'Active devices: 3  (Speaker=2, SmartPlug=1)'."""

    def __init__(self, active: Callable[[], Mapping[str, set[str]]]):
        self.active = active
        self.label: tk.Label | None = None

    def mount(self, parent: tk.Misc, status: StatusSetter) -> None:
        self.label = tk.Label(parent, bg=PAGE, fg=HEADER, font=("Segoe UI", 9), pady=4)
        self.label.pack(fill="x")
        self.refresh()

    def refresh(self) -> None:
        active = {k: v for k, v in self.active().items() if v}
        parts = ", ".join(f"{k}={len(v)}" for k, v in active.items())
        total = sum(len(v) for v in active.values())
        self.label.config(text=f"Active devices: {total}" + (f"  ({parts})" if parts else ""))


class TriggerAll(Widget):
    def __init__(self, control: ControlState, on_change: Callable[[], None]):
        self.control = control
        self.on_change = on_change  # lets the grid repaint at once
        self.button: tk.Button | None = None
        self._status: StatusSetter | None = None

    def mount(self, parent: tk.Misc, status: StatusSetter) -> None:
        self._status = status
        frame = tk.Frame(parent, bg=PAGE)
        frame.pack(fill="x", padx=10, pady=(10, 4))
        self.button = tk.Button(frame, fg="white", relief="flat", font=_BOLD, height=2, command=self.click)
        self.button.pack(fill="x")
        self.refresh()

    def click(self) -> None:
        on = not self.control.hijacked
        self.control.set_all(on)
        self.on_change()
        self.refresh()
        self._status(stamped(f"All hijacks {'ENABLED' if on else 'DISABLED'}"), HIJACK if on else OFF)

    def refresh(self) -> None:
        on = bool(self.control.hijacked)
        self.button.config(
            text="TRIGGER ALL: ON  (click to clear)" if on else "TRIGGER ALL: OFF  (click to enable all)",
            bg=HIJACK if on else OFF,
            activebackground=HIJACK_HOVER if on else OFF_HOVER,
        )


class StreamGrid(Widget):
    """Per camera: a status dot (hijacked / no picture / its zone's devices
    on / idle), a hijack button, and optionally a recording button."""

    def __init__(
        self,
        control: ControlState,
        labels: list[str],
        devices_of: Mapping[int, set[str]],
        active_devices: Callable[[], set[str]],
        recording: Callable[[int], Toggle] | None = None,
        offline: Callable[[], set[int]] | None = None,
        columns: int = 4,
    ):
        self.control = control
        self.labels = labels
        self.devices_of = devices_of
        self.active_devices = active_devices
        self.recording = recording
        self.offline = offline
        self.columns = columns
        self.dots: dict[int, tk.Label] = {}
        self.buttons: dict[int, tk.Button] = {}
        self.rec: dict[int, tuple[tk.Button, Toggle]] = {}
        self._status: StatusSetter | None = None

    def mount(self, parent: tk.Misc, status: StatusSetter) -> None:
        self._status = status
        box = tk.Frame(parent, bg=PAGE)
        box.pack(fill="both", expand=True, padx=10, pady=10)
        canvas = tk.Canvas(box, bg=PAGE, highlightthickness=0, height=420)
        bar = tk.Scrollbar(box, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        grid = tk.Frame(canvas, bg=PAGE)
        canvas.create_window((0, 0), window=grid, anchor="nw")
        grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        for sid, label in enumerate(self.labels):
            cell = tk.Frame(grid, bg=PAGE, padx=5, pady=5)
            cell.grid(row=sid // self.columns, column=sid % self.columns)
            self.dots[sid] = tk.Label(cell, text="●", fg=DOT_IDLE, bg=PAGE, font=("Segoe UI", 10))
            self.dots[sid].pack()
            self.buttons[sid] = tk.Button(
                cell,
                text=label,
                width=16,
                height=3,
                relief="flat",
                font=("Segoe UI", 9),
                command=lambda s=sid: self.click(s),
            )
            self.buttons[sid].pack()
            if self.recording is not None:
                toggle = self.recording(sid)
                button = tk.Button(
                    cell,
                    width=16,
                    height=1,
                    relief="flat",
                    font=("Segoe UI", 9, "bold"),
                    command=lambda s=sid: self.click_rec(s),
                )
                button.pack(pady=(2, 0))
                self.rec[sid] = (button, toggle)
        self.refresh()

    def click(self, sid: int) -> None:
        on = self.control.toggle(sid)
        self.refresh()
        self._status(stamped(f"Camera {sid} hijack {'ENABLED' if on else 'DISABLED'}"), HIJACK if on else OFF)

    def click_rec(self, sid: int) -> None:
        _, toggle = self.rec[sid]
        on = not toggle.get()
        toggle.set(on)
        self.refresh()
        self._status(stamped(f"Camera {sid} recording {'ON' if on else 'OFF'}"), ON if on else OFF)

    def refresh(self) -> None:
        active = self.active_devices()
        offline = self.offline() if self.offline is not None else set()
        for sid, dot in self.dots.items():
            hijacked = sid in self.control.hijacked
            if hijacked:
                color = HIJACK
            elif sid in offline:
                color = DOT_OFFLINE
            else:
                color = DOT_ACTIVE if self.devices_of.get(sid, set()) & active else DOT_IDLE
            dot.config(fg=color)
            self.buttons[sid].config(bg=HIJACK if hijacked else "#ffffff", fg="white" if hijacked else HEADER)
        for button, toggle in self.rec.values():
            on = toggle.get()
            button.config(text="REC ●" if on else "REC", bg=ON if on else "#ffffff", fg="white" if on else HEADER)
