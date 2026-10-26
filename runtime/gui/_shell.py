"""Shared base class for the operator and debug GUI shells.

Both shells own a Tk root + a vertical stack of GuiComponents plus a header
and a status bar. Only `setup()` differs per shell (operator uses a fresh
`tk.Tk`; debug attaches as a `tk.Toplevel` when given a master) — everything
else lives here.
"""

import logging
import time
import tkinter as tk
from typing import Optional

from ._base import GuiComponent
from ._style import BG_HEADER, BG_PAGE


logger = logging.getLogger(__name__)


class _GuiShell:
    """Common lifecycle + builders shared by InteractionGUI and ManualTriggerGUI."""

    REFRESH_HZ = 4.0

    def __init__(
        self,
        components: list[GuiComponent],
        title: str,
        header_text: str,
        name: str,
    ):
        self.components = components
        self.title = title
        self.header_text = header_text
        self.name = name

        self.root: Optional[tk.Misc] = None
        self.status_label: Optional[tk.Label] = None

        self._last_refresh = 0.0
        self._refresh_interval = 1.0 / self.REFRESH_HZ

    # ------------------------------------------------------------------
    # Lifecycle (subclasses override `_create_root`)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        try:
            self.root = self._create_root()
        except tk.TclError as e:
            logger.error("[%s] Tkinter unavailable: %s", self.name, e)
            self.root = None
            return
        if self.root is None:
            return

        self.root.title(self.title)
        try:
            self.root.attributes("-topmost", True)
        except tk.TclError:
            pass
        self.root.configure(bg=BG_PAGE)
        self._apply_geometry()

        # Order matters: build status bar before body so set_status is live
        # if a component calls it from `mount`.
        self._build_header()
        self._build_status_bar()
        self._build_body()

    def _create_root(self) -> tk.Misc:
        """Construct the Tk root (or Toplevel). Subclasses must override."""
        raise NotImplementedError

    def _apply_geometry(self) -> None:
        """Hook for shells that want a fixed size/position. Default = no-op."""

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
                        logger.debug("[%s] %s refresh error: %s", self.name, type(comp).__name__, e)
                self._last_refresh = now
            self._pump_events()
        except tk.TclError:
            self.root = None

    def _pump_events(self) -> None:
        """Drain the Tk event loop. Toplevel children are pumped by the master."""
        self.root.update()

    def destroy(self) -> None:
        if not self.root:
            return
        try:
            logger.info("[%s] Destroying GUI...", self.name)
            self.root.destroy()
        except Exception as e:
            logger.error("[%s] Error during destroy: %s", self.name, e, exc_info=True)
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
                logger.error("[%s] %s mount failed: %s", self.name, type(comp).__name__, e, exc_info=True)
