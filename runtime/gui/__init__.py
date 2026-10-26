"""Public surface of the GUI package.

Two shells (`InteractionGUI`, `ManualTriggerGUI`) plus the three handler
Protocols those shells type their handler params against. Consumers that
implement a handler (e.g. ResultGUIController, MonitorHandler) should import
the Protocols from here, not from the underscored `_base` module.
"""

from ._base import DensityMapTogglable, MonitorTogglable, RecorderTogglable
from .interaction_gui import InteractionGUI
from .manual_gui import ManualTriggerGUI


__all__ = [
    "DensityMapTogglable",
    "InteractionGUI",
    "ManualTriggerGUI",
    "MonitorTogglable",
    "RecorderTogglable",
]
