"""Self-contained widgets. Add a new feature = add a new file + append to a factory."""

from .activity_label import ActivityLabel
from .cancel_all import CancelAllButton
from .monitor_toggle import MonitorToggleButton
from .recorder_toggle import RecorderToggleButton
from .stream_grid import StreamGridComponent
from .terminate_button import TerminateButton
from .trigger_all import TriggerAllButton


__all__ = [
    "ActivityLabel",
    "CancelAllButton",
    "MonitorToggleButton",
    "RecorderToggleButton",
    "StreamGridComponent",
    "TerminateButton",
    "TriggerAllButton",
]
