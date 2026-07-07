"""Debug GUI shell. Same component-stack pattern as InteractionGUI; attaches as a
Toplevel when a `master` is supplied so we keep a single Tcl interpreter."""

import logging
import tkinter as tk
from collections.abc import Callable
from typing import Optional

from runtime.config import Config

from ._base import DensityMapTogglable, GuiComponent, RecorderTogglable
from ._shell import _GuiShell
from .components import (
    DensityMapToggleButton,
    RecorderToggleButton,
    StreamGridComponent,
    TriggerAllToggleButton,
)


logger = logging.getLogger(__name__)


class ManualTriggerGUI(_GuiShell):
    """Debug-only manual hijack panel.

    Hosts a stack of GuiComponents — same lifecycle contract as InteractionGUI.
    Exposes `reset_all_hijacks()` for callers that need to clear the grid from
    outside (e.g., the Cancel All button on the operator GUI).
    """

    def __init__(
        self,
        components: list[GuiComponent],
        title: str = "DEBUG - Manual Hijack",
        header_text: str = "DEBUG - MANUAL HIJACK",
        master: Optional[tk.Misc] = None,
        name: str = "TriggerGUI",
        grid: Optional[StreamGridComponent] = None,
    ):
        super().__init__(components, title, header_text, name)
        self.master = master
        self._grid = grid

    def _create_root(self) -> tk.Misc:
        return tk.Toplevel(self.master) if self.master is not None else tk.Tk()

    def _pump_events(self) -> None:
        # Toplevel children are pumped by the master Tk's update() call.
        if self.master is None:
            self.root.update()

    # ------------------------------------------------------------------
    # External hook
    # ------------------------------------------------------------------

    def reset_all_hijacks(self) -> None:
        """Called after an external Cancel All to wipe the visual hijack state."""
        if self._grid is not None:
            self._grid.reset_hijacks()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def debug_panel(
        cls,
        *,
        config: Config,
        on_trigger: Callable[[int], None],
        on_trigger_all: Optional[Callable[[bool], None]] = None,
        recorder_handler: Optional[RecorderTogglable] = None,
        density_map_handler: Optional[DensityMapTogglable] = None,
        status_provider: Optional[Callable[[], dict]] = None,
        master: Optional[tk.Misc] = None,
    ) -> "ManualTriggerGUI":
        """Standard debug panel: trigger-all + recorder + density-map + stream grid.

        Optional pieces are skipped when their callback/handler is None.
        """
        grid = StreamGridComponent(
            config=config,
            on_trigger=on_trigger,
            recorder_handler=recorder_handler,
            status_provider=status_provider,
        )

        components: list[GuiComponent] = []
        if on_trigger_all is not None:
            components.append(TriggerAllToggleButton(on_trigger_all=on_trigger_all, grid=grid))
        if recorder_handler is not None:
            components.append(RecorderToggleButton(handler=recorder_handler, grid=grid))
        if density_map_handler is not None:
            components.append(DensityMapToggleButton(handler=density_map_handler))
        components.append(grid)

        return cls(components=components, master=master, grid=grid)
