"""The panels' widgets on a real (hidden) Tk root; skipped where Tk has no display.

Tk is created with pytest's output capture off: under that capture, Tcl on
Windows intermittently fails to find its own library ("Can't find a usable
init.tcl"). Plain stdout/stderr redirection does not trigger it.
"""

import tkinter as tk

import pytest

from birdcount.deploy.gui import widgets
from birdcount.deploy.gui.controls import ControlState, Toggle
from birdcount.deploy.gui.panels import Panels, debug_panel, operator_panel
from birdcount.deploy.gui.widgets import ActivityLabel, ConfirmButton, StreamGrid, ToggleButton, TriggerAll


@pytest.fixture
def root(capfd):
    try:
        with capfd.disabled():
            r = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no Tk display: {e}")
    r.withdraw()
    yield r
    r.destroy()


class Status:
    def __init__(self):
        self.text = None

    def __call__(self, text, color=None):
        self.text = text


class Switch:
    def __init__(self, on=False):
        self.on = on

    def toggle(self, label="Monitor") -> Toggle:
        return Toggle(label, lambda: self.on, self.set)

    def set(self, on: bool) -> None:
        self.on = on


def test_toggle_button_flips_and_follows_outside_changes(root) -> None:
    switch, status = Switch(), Status()
    button = ToggleButton(switch.toggle())
    button.mount(root, status)
    assert "OFF" in button.button.cget("text").split("(")[0]
    button.click()
    assert switch.on and "Monitor turned ON" in status.text
    assert button.button.cget("bg") == widgets.ON
    switch.on = False  # switched elsewhere
    button.refresh()
    assert button.button.cget("bg") == widgets.OFF


def test_confirm_button_acts_only_when_confirmed(root, monkeypatch) -> None:
    calls, status = [], Status()
    button = ConfirmButton("TERMINATE PROGRAM", "Sure?", lambda: calls.append(1), "Shutting down", once=True)
    button.mount(root, status)
    monkeypatch.setattr(widgets.messagebox, "askyesno", lambda *a, **k: False)
    button.click()
    assert calls == [] and status.text is None
    monkeypatch.setattr(widgets.messagebox, "askyesno", lambda *a, **k: True)
    button.click()
    assert calls == [1] and status.text.startswith("Shutting down")
    assert button.button.cget("state") == "disabled"


def test_stream_grid_and_trigger_all_share_one_control_state(root) -> None:
    control, status = ControlState(3), Status()
    recording = {sid: Switch() for sid in range(3)}
    active = set()
    grid = StreamGrid(
        control, ["A", "B", "C"], {0: {"spk1"}}, lambda: active, lambda sid: recording[sid].toggle("REC")
    )
    trigger = TriggerAll(control, grid.refresh)
    grid.mount(root, status)
    trigger.mount(root, status)

    grid.click(1)
    assert control.hijacked == {1}
    assert grid.dots[1].cget("fg") == widgets.HIJACK
    active.add("spk1")  # camera 0's zone device switched on by the alarm
    grid.refresh()
    assert grid.dots[0].cget("fg") == widgets.DOT_ACTIVE
    assert grid.dots[2].cget("fg") == widgets.DOT_IDLE

    trigger.click()  # some hijacked -> clear all
    assert control.hijacked == set()
    assert grid.buttons[1].cget("bg") == "#ffffff"
    trigger.click()
    assert control.hijacked == {0, 1, 2}

    grid.click_rec(2)
    assert recording[2].on and grid.rec[2][0].cget("text") == "REC ●"


def test_activity_label_counts_active_devices(root) -> None:
    label = ActivityLabel(lambda: {"Speaker": {"a", "b"}, "SmartPlug": set()})
    label.mount(root, Status())
    assert label.label.cget("text") == "Active devices: 2  (Speaker=2)"


def test_panels_open_pump_and_close(capfd) -> None:
    control = ControlState(2)
    panels = Panels(
        operator_panel(
            on_cancel_all=lambda: None,
            on_terminate=lambda: None,
            monitor=Switch().toggle(),
            active=lambda: {},
            screen_width=1920,
        ),
        debug_panel(
            control=control,
            labels=["A", "B"],
            devices_of={},
            active_devices=set,
            toggles=[Switch().toggle("Heatmap")],
            recording=None,
        ),
    )
    with capfd.disabled():
        if not panels.open():
            pytest.skip("no Tk display")
    try:
        panels.operator.root.withdraw()
        panels.debug.root.withdraw()
        panels.pump()
        assert panels.operator.root is not None
    finally:
        panels.close()
    assert panels.operator.root is None and panels.debug.root is None
    panels.pump()  # closed: a no-op
