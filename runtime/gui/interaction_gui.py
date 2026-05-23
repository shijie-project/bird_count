"""Operator-facing GUI shell. Owns a Tk root + a stack of GuiComponents."""

import logging
import tkinter as tk
from collections.abc import Callable

from ._base import GuiComponent, MonitorTogglable
from ._shell import _GuiShell
from .components import ActivityLabel, CancelAllButton, MonitorToggleButton, TerminateButton


logger = logging.getLogger(__name__)


class InteractionGUI(_GuiShell):
    """Thin operator shell: header + body (vertical component stack) + status bar.

    Adding a new control = define a `GuiComponent` and append it to `components`.
    The shell stays agnostic about what each component does.
    """

    def __init__(
        self,
        components: list[GuiComponent],
        title: str = "Bird Count - Operator Control",
        header_text: str = "OPERATOR CONTROL",
        name: str = "InteractionGUI",
    ):
        super().__init__(components, title, header_text, name)

    def _create_root(self) -> tk.Misc:
        return tk.Tk()

    def _apply_geometry(self) -> None:
        # Place top-right so it doesn't collide with the debug GUI.
        try:
            sw = self.root.winfo_screenwidth()
            self.root.geometry(f"360x400+{max(0, sw - 380)}+20")
            self.root.resizable(False, False)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def operator_panel(
        cls,
        *,
        on_cancel_all: Callable[[], None],
        on_terminate: Callable[[], None] | None = None,
        monitor_handler: MonitorTogglable | None = None,
        active_devices_provider: Callable[[], dict] | None = None,
    ) -> "InteractionGUI":
        """Standard operator panel: Cancel All + (optional) Monitor toggle +
        (optional) activity label + (optional) Terminate Program button."""
        components: list[GuiComponent] = [CancelAllButton(on_cancel_all=on_cancel_all)]
        if monitor_handler is not None:
            components.append(MonitorToggleButton(handler=monitor_handler))
        if active_devices_provider is not None:
            components.append(ActivityLabel(status_provider=active_devices_provider))
        if on_terminate is not None:
            components.append(TerminateButton(on_terminate=on_terminate))
        return cls(components)
