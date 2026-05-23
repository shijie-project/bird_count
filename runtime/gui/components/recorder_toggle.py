"""Global ON/OFF button for the recorder handler; cascades visual state to the grid."""

from .._base import RecorderTogglable
from ._bicolor_toggle import _BicolorToggleButton
from .stream_grid import StreamGridComponent


class RecorderToggleButton(_BicolorToggleButton):
    """Click flips recorder.toggle() then refreshes every per-stream REC button."""

    def __init__(self, handler: RecorderTogglable, grid: StreamGridComponent):
        super().__init__(
            label="RECORDING",
            toggle_fn=handler.toggle,
            is_enabled_fn=handler.is_enabled,
            log_tag="RecorderToggleButton",
            on_state_change=lambda _state: grid.refresh_rec_buttons(),
            padx=10,
            pady=0,
        )
