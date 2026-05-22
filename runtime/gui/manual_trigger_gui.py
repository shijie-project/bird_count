"""Debug GUI shell. Same component-stack pattern as InteractionGUI; attaches as a
Toplevel when a `master` is supplied so we keep a single Tcl interpreter."""

import logging
import time
import tkinter as tk
from typing import Callable, Optional

from runtime.config import Config

from ._base import DensityMapTogglable, GuiComponent, RecorderHandler
from ._style import BG_HEADER, BG_PAGE
from .components import (
    DensityMapToggleButton,
    RecorderToggleButton,
    StreamGridComponent,
    TriggerAllToggleButton,
)


logger = logging.getLogger(__name__)


class ManualTriggerGUI:
    """Debug-only manual hijack panel.

    Hosts a stack of GuiComponents — same lifecycle contract as InteractionGUI.
    Exposes `reset_all_hijacks()` for callers that need to clear the grid from
    outside (e.g., the Cancel All button on the operator GUI).
    """

    REFRESH_HZ = 4.0

    def __init__(
        self,
        components: list[GuiComponent],
        title: str = "DEBUG - Manual Hijack",
        header_text: str = "DEBUG - MANUAL HIJACK",
        master: Optional[tk.Misc] = None,
        name: str = "TriggerGUI",
    ):
        self.components = components
        self.title = title
        self.header_text = header_text
        self.master = master
        self.name = name

        self.root: Optional[tk.Misc] = None
        self.status_label: Optional[tk.Label] = None

        self._last_refresh = 0.0
        self._refresh_interval = 1.0 / self.REFRESH_HZ

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        try:
            self.root = tk.Toplevel(self.master) if self.master is not None else tk.Tk()
        except tk.TclError as e:
            logger.error(f"[{self.name}] Tkinter unavailable: {e}")
            self.root = None
            return

        self.root.title(self.title)
        try:
            self.root.attributes("-topmost", True)
        except tk.TclError:
            pass
        self.root.configure(bg=BG_PAGE)

        self._build_header()
        self._build_body()
        self._build_status_bar()

    def update(self) -> None:
        if not self.root:
            return
        try:
            now = time.time()
            if now - self._last_refresh >= self._refresh_interval:
                for comp in self.components:
                    try:
                        comp.refresh()
                    except Exception as e:
                        logger.debug(f"[{self.name}] {type(comp).__name__} refresh error: {e}")
                self._last_refresh = now
            # Toplevel children are pumped by the master Tk's update() call.
            if self.master is None:
                self.root.update()
        except tk.TclError:
            self.root = None

    def destroy(self) -> None:
        if not self.root:
            return
        try:
            logger.info(f"[{self.name}] Destroying manual trigger GUI...")
            self.root.destroy()
        except Exception as e:
            logger.error(f"[{self.name}] Error during destroy: {e}")
        finally:
            self.root = None

    # ------------------------------------------------------------------
    # External hook
    # ------------------------------------------------------------------

    def reset_all_hijacks(self) -> None:
        """Called after an external Cancel All to wipe the visual hijack state."""
        for c in self.components:
            if isinstance(c, StreamGridComponent):
                c.reset_hijacks()
                return

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def set_status(self, text: str, fg: Optional[str] = None) -> None:
        if not self.status_label:
            return
        try:
            kwargs: dict = {"text": text}
            if fg is not None:
                kwargs["fg"] = fg
            self.status_label.config(**kwargs)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG_HEADER, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text=self.header_text,
            fg="white",
            bg=BG_HEADER,
            font=("Segoe UI", 12, "bold"),
        ).pack()

    def _build_status_bar(self) -> None:
        self.status_label = tk.Label(
            self.root,
            text="Ready...",
            bd=1,
            relief="sunken",
            anchor="w",
            font=("Segoe UI", 8),
            padx=10,
            pady=5,
        )
        self.status_label.pack(fill="x", side="bottom")

    def _build_body(self) -> None:
        # Components pack themselves top-down; pinned status bar lives below.
        body = tk.Frame(self.root, bg=BG_PAGE)
        body.pack(fill="both", expand=True)
        for comp in self.components:
            try:
                comp.mount(body, set_status=self.set_status)
            except Exception as e:
                logger.error(f"[{self.name}] {type(comp).__name__} mount failed: {e}")

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
        recorder_handler: Optional[RecorderHandler] = None,
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

        return cls(components=components, master=master)
