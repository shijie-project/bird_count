"""Component ABC + duck-typed handler protocols.

Components mount themselves into a parent frame and optionally refresh on a
periodic tick. Host shells (InteractionGUI / ManualTriggerGUI) own the Tk
root and a list of components — nothing else.
"""

import tkinter as tk
from abc import ABC, abstractmethod
from typing import Protocol

from ._style import StatusSetter


class GuiComponent(ABC):
    """Self-contained widget. One class = one feature."""

    @abstractmethod
    def mount(self, parent: tk.Misc, set_status: StatusSetter | None = None) -> None:
        """Build and pack widgets into `parent`. Called once during shell setup."""

    def refresh(self) -> None:
        """Periodic hook driven by the host's update loop. Default = no-op."""


class MonitorTogglable(Protocol):
    """Contract for the handler driving MonitorToggleButton."""

    def toggle(self) -> bool: ...
    def is_enabled(self) -> bool: ...


class RecorderTogglable(Protocol):
    """Contract for the recorder handler used by RecorderToggleButton + StreamGridComponent."""

    def toggle(self) -> bool: ...
    def is_enabled(self) -> bool: ...
    def toggle_stream(self, stream_id: int) -> bool: ...
    def is_stream_enabled(self, stream_id: int) -> bool: ...


class DensityMapTogglable(Protocol):
    """Contract for the handler driving DensityMapToggleButton.

    Uses qualified method names (`toggle_density_map` / `is_density_map_enabled`)
    because the monitor handler implements both this protocol and
    `MonitorTogglable` on the same class — generic `toggle()` / `is_enabled()`
    names would collide. Bicolor-toggle subclasses bind the right pair of
    callables internally, so the asymmetry doesn't leak into the UI base.
    """

    def toggle_density_map(self) -> bool: ...
    def is_density_map_enabled(self) -> bool: ...
