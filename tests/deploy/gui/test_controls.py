"""ControlState: the one set of hand-hijacked cameras, and what it audits."""

from birdcount.deploy.gui.controls import ControlState


class Audit:
    def __init__(self):
        self.events = []

    def log(self, event, **fields):
        self.events.append((event, fields.get("stream_id")))


def test_toggle_and_trigger_all_audit_each_camera_once() -> None:
    audit = Audit()
    control = ControlState(3, audit)
    assert control.toggle(1) is True
    assert control.hijacked == {1}
    control.set_all(True)
    assert control.hijacked == {0, 1, 2}
    assert audit.events == [("hijack.enable", 1), ("hijack.enable", 0), ("hijack.enable", 2)]
    assert control.toggle(2) is False
    control.set_all(False)
    assert control.hijacked == set()
    assert audit.events[3:] == [("hijack.disable", 2), ("hijack.disable", 0), ("hijack.disable", 1)]


def test_clear_returns_what_was_hijacked() -> None:
    control = ControlState(4)
    control.toggle(3)
    control.toggle(0)
    assert control.clear() == [0, 3]
    assert control.hijacked == set()
