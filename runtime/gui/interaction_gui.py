"""Operator-facing GUI shell. Owns a Tk root + a stack of GuiComponents."""

import logging
import time
import tkinter as tk
from typing import Callable, Optional

from ._base import GuiComponent, MonitorTogglable
from ._style import BG_HEADER, BG_PAGE
from .components import ActivityLabel, CancelAllButton, MonitorToggleButton, TerminateButton


logger = logging.getLogger(__name__)


class InteractionGUI:
    """Thin operator shell: header + body (vertical component stack) + status bar.

    Adding a new control = define a `GuiComponent` and append it to `components`.
    The shell stays agnostic about what each component does.
    """

    REFRESH_HZ = 4.0

    def __init__(
        self,
        components: list[GuiComponent],
        title: str = "Bird Count - Operator Control",
        header_text: str = "OPERATOR CONTROL",
        name: str = "InteractionGUI",
    ):
        self.components = components
        self.title = title
        self.header_text = header_text
        self.name = name

        self.root: Optional[tk.Tk] = None
        self.status_label: Optional[tk.Label] = None

        self._last_refresh = 0.0
        self._refresh_interval = 1.0 / self.REFRESH_HZ

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            logger.error(f"[{self.name}] Tkinter unavailable: {e}")
            self.root = None
            return

        self.root.title(self.title)
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.configure(bg=BG_PAGE)

        # Place top-right so it doesn't collide with the debug GUI.
        try:
            sw = self.root.winfo_screenwidth()
            self.root.geometry(f"360x400+{max(0, sw - 380)}+20")
        except tk.TclError:
            pass

        self._build_header()
        self._build_status_bar()  # before body, so set_status is live during mount
        self._build_body()

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
            self.root.update()
        except tk.TclError:
            self.root = None

    def destroy(self) -> None:
        if not self.root:
            return
        try:
            logger.info(f"[{self.name}] Destroying interaction GUI...")
            self.root.destroy()
        except Exception as e:
            logger.error(f"[{self.name}] Error during destroy: {e}")
        finally:
            self.root = None

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def set_status(self, text: str, fg: Optional[str] = None) -> None:
        """Components call this through their set_status kwarg at mount time."""
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
        header = tk.Frame(self.root, bg=BG_HEADER)
        header.pack(pady=10, fill="x")
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
            pady=4,
        )
        self.status_label.pack(fill="x", side="bottom")

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=BG_PAGE)
        body.pack(fill="both", expand=True)
        for comp in self.components:
            try:
                comp.mount(body, set_status=self.set_status)
            except Exception as e:
                logger.error(f"[{self.name}] {type(comp).__name__} mount failed: {e}")

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def operator_panel(
        cls,
        *,
        on_cancel_all: Callable[[], None],
        on_terminate: Optional[Callable[[], None]] = None,
        monitor_handler: Optional[MonitorTogglable] = None,
        active_devices_provider: Optional[Callable[[], dict]] = None,
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
