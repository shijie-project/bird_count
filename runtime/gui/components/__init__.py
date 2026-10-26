"""Self-contained widgets. Add a new feature = add a new file + append to a factory."""

from .activity_label import ActivityLabel
from .cancel_all_button import CancelAllButton
from .density_map_toggle import DensityMapToggleButton
from .monitor_toggle import MonitorToggleButton
from .recorder_toggle import RecorderToggleButton
from .stream_grid import StreamGridComponent
from .terminate_button import TerminateButton
from .trigger_all_toggle import TriggerAllToggleButton


__all__ = [
    "ActivityLabel",
    "CancelAllButton",
    "DensityMapToggleButton",
    "MonitorToggleButton",
    "RecorderToggleButton",
    "StreamGridComponent",
    "TerminateButton",
    "TriggerAllToggleButton",
]
