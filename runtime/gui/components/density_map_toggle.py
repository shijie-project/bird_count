"""Bicolor ON/OFF button bound to a DensityMapTogglable handler."""

from .._base import DensityMapTogglable
from ._bicolor_toggle import _BicolorToggleButton


class DensityMapToggleButton(_BicolorToggleButton):
    """Click flips density-map overlay via `handler.toggle_density_map()`;
    re-syncs each refresh so external state changes show up even without a click."""

    def __init__(self, handler: DensityMapTogglable):
        super().__init__(
            label="DENSITY MAP",
            toggle_fn=handler.toggle_density_map,
            is_enabled_fn=handler.is_density_map_enabled,
            log_tag="DensityMapToggleButton",
            padx=10,
            pady=(0, 4),
        )
