"""The two panels, assembled, and the loop hook that keeps them alive.

Operator panel (always): Cancel All, the monitor switch, active devices,
Terminate. Debug panel (DEBUG on): Trigger All, recording / overlay switches,
and the camera grid with a hijack button per camera.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Mapping

from birdcount.deploy.gui.controls import ControlState, Toggle
from birdcount.deploy.gui.shell import Panel
from birdcount.deploy.gui.widgets import ActivityLabel, ConfirmButton, StreamGrid, ToggleButton, TriggerAll


def operator_panel(
    *,
    on_cancel_all: Callable[[], None],
    on_terminate: Callable[[], None],
    monitor: Toggle | None,
    active: Callable[[], Mapping[str, set[str]]],
    screen_width: int,
) -> Panel:
    widgets = [
        ConfirmButton(
            "CANCEL ALL ALERTS",
            "Cancel all currently active alerts?\n\nThis turns off every device the alarm switched on.",
            on_cancel_all,
            "All alerts cancelled",
        )
    ]
    if monitor is not None:
        widgets.append(ToggleButton(monitor))
    widgets += [
        ActivityLabel(active),
        ConfirmButton(
            "TERMINATE PROGRAM",
            "Shut down the whole bird-count program?\n\nCapture, inference, recording, alerts and this "
            "panel stop; recordings are finalised cleanly.",
            on_terminate,
            "Shutting down...",
            once=True,
        ),
    ]
    # Top-right, clear of the debug panel.
    return Panel(
        "Bird Count - Operator", "OPERATOR CONTROL", widgets, geometry=f"360x400+{max(0, screen_width - 380)}+20"
    )


def debug_panel(
    *,
    control: ControlState,
    labels: list[str],
    devices_of: Mapping[int, set[str]],
    active_devices: Callable[[], set[str]],
    toggles: list[Toggle],
    recording: Callable[[int], Toggle] | None,
) -> Panel:
    grid = StreamGrid(control, labels, devices_of, active_devices, recording)
    widgets = [TriggerAll(control, grid.refresh), *(ToggleButton(t) for t in toggles), grid]
    return Panel("Bird Count - Debug", "DEBUG - MANUAL HIJACK", widgets)


class Panels:
    """Both windows on one Tk interpreter, pumped from the consumer loop."""

    def __init__(self, operator: Panel, debug: Panel | None = None):
        self.operator, self.debug = operator, debug

    def open(self) -> bool:
        if not self.operator.open():
            return False
        if self.debug is not None:
            self.debug.open(master=self.operator.root)
        return True

    def pump(self) -> None:
        """Process window events and refresh widgets; never blocks."""
        if self.operator.root is None:
            return
        try:
            self.operator.refresh()
            if self.debug is not None:
                self.debug.refresh()
            self.operator.root.update()
        except tk.TclError:  # the window was destroyed
            self.operator.root = None

    def close(self) -> None:
        if self.debug is not None:
            self.debug.close()
        self.operator.close()
