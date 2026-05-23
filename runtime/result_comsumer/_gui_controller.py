"""ResultGUIController — encapsulates everything GUI-related in the consumer process.

This is the only file in `result_comsumer/` that knows about `runtime.gui`. It
owns:
  * the operator + debug Tk windows,
  * the manual-hijack state the alert evaluator reads from,
  * every widget callback (manual trigger, trigger-all, cancel-all, terminate).

The consumer's process-side stays narrow: it constructs the controller once,
calls `pump()` per tick, and `destroy()` on shutdown. Cross-component coupling
flows through explicit callbacks (`on_cancel_all`, `on_terminate`) and
read-only providers, never via shared mutable state beyond
`manual_override_streams` (intentionally exposed for the alert evaluator).
"""

import logging
from collections.abc import Callable
from typing import Optional

from runtime.audit import AuditLog
from runtime.config import Config
from runtime.gui._base import DensityMapTogglable, MonitorTogglable, RecorderTogglable


logger = logging.getLogger(__name__)


class ResultGUIController:
    """Owns the consumer process's Tk windows + the manual-hijack state.

    All methods run on the consumer's main thread — the thread that pumps Tk
    via `pump()`. Widget callbacks therefore see a stable, single-threaded
    view of every attribute; no locking required.

    The `audit` and handler references are read-only borrows from
    `ResultProcess`. Pass the *opened* AuditLog (not the disabled placeholder)
    so widget callbacks land in the right log file.
    """

    def __init__(
        self,
        *,
        config: Config,
        audit: AuditLog,
        debug_mode: bool,
        on_cancel_all: Callable[[], None],
        on_terminate: Callable[[], None],
        monitor_handler: Optional[MonitorTogglable] = None,
        recorder_handler: Optional[RecorderTogglable] = None,
        density_map_handler: Optional[DensityMapTogglable] = None,
        active_devices_provider: Optional[Callable[[], dict]] = None,
        name: str = "ResultGUI",
    ):
        self.config = config
        self.audit = audit
        self.debug_mode = debug_mode
        self.name = name

        self._on_cancel_all = on_cancel_all
        self._on_terminate = on_terminate
        self._monitor_handler = monitor_handler
        self._recorder_handler = recorder_handler
        self._density_map_handler = density_map_handler
        self._active_devices_provider = active_devices_provider

        # Shared with the consumer thread's alert evaluator — read by
        # `_AlertEvaluator.evaluate_inplace`, mutated only by widget callbacks.
        self.manual_override_streams: set[int] = set()

        # Tk handles. Populated by `setup()`; either may remain None if the
        # runtime.gui import fails or debug_mode is off.
        self._operator_gui = None
        self._debug_gui = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def setup(self) -> None:
        """Mount the operator GUI (always) + debug GUI (when debug_mode).

        Tolerates a missing `runtime.gui` module (headless deployments) and
        any Tcl error during construction — in both cases the windows simply
        don't appear and the consumer keeps running.
        """
        try:
            from runtime.gui import InteractionGUI, ManualTriggerGUI
        except ImportError:
            logger.warning(f"[{self.name}] runtime.gui not available; running headless.")
            return

        try:
            self._operator_gui = InteractionGUI.operator_panel(
                on_cancel_all=self._handle_cancel_all,
                on_terminate=self._handle_terminate,
                monitor_handler=self._monitor_handler,
                active_devices_provider=self._active_devices_provider,
            )
            self._operator_gui.setup()

            if not self.debug_mode:
                return

            # Debug panel reuses the operator's Tk root as a Toplevel so a
            # single Tcl interpreter drives both windows.
            self._debug_gui = ManualTriggerGUI.debug_panel(
                config=self.config,
                master=self._operator_gui.root,
                on_trigger=self._handle_manual_trigger,
                on_trigger_all=self._handle_trigger_all,
                recorder_handler=self._recorder_handler,
                density_map_handler=self._density_map_handler,
                status_provider=self._active_devices_provider,
            )
            self._debug_gui.setup()
        except Exception as e:
            logger.warning(f"[{self.name}] GUI initialization failed: {e}")
            self._operator_gui = None
            self._debug_gui = None

    def pump(self) -> None:
        """Service Tk events. Called every consumer-loop iteration."""
        if self._operator_gui is not None:
            self._operator_gui.update()
        if self._debug_gui is not None:
            self._debug_gui.update()

    def destroy(self) -> None:
        """Tear Tk down. Debug Toplevel goes before its master root."""
        if self._debug_gui is not None:
            try:
                self._debug_gui.destroy()
            except Exception as e:
                logger.error(f"[{self.name}] debug GUI destroy failed: {e}")
            self._debug_gui = None
        if self._operator_gui is not None:
            try:
                self._operator_gui.destroy()
            except Exception as e:
                logger.error(f"[{self.name}] operator GUI destroy failed: {e}")
            self._operator_gui = None

    # ==================================================================
    # Widget callbacks
    # ==================================================================

    def _handle_manual_trigger(self, stream_id: int) -> None:
        """Toggle the manual-hijack flag for one stream."""
        if stream_id in self.manual_override_streams:
            self.manual_override_streams.discard(stream_id)
            self.audit.log("hijack.disable", stream_id=stream_id)
            logger.info(f"[{self.name}] Manual Hijack DISABLED for Stream {stream_id}.")
        else:
            self.manual_override_streams.add(stream_id)
            self.audit.log("hijack.enable", stream_id=stream_id)
            logger.info(f"[{self.name}] Manual Hijack ENABLED for Stream {stream_id}.")

    def _handle_trigger_all(self, target: bool) -> None:
        """Force every known stream into `target` hijack state."""
        all_sids = set(self.config.sid_to_ip.keys())
        if target:
            affected = sorted(all_sids - self.manual_override_streams)
            self.manual_override_streams |= all_sids
            event, verb = "hijack.enable", "ENABLED"
        else:
            affected = sorted(self.manual_override_streams)
            self.manual_override_streams.clear()
            event, verb = "hijack.disable", "DISABLED"

        for sid in affected:
            self.audit.log(event, stream_id=sid, source="trigger_all")
        logger.info(f"[{self.name}] TRIGGER ALL: {verb} hijack on {len(affected)} stream(s): {affected}")

    def _handle_cancel_all(self) -> None:
        """Operator 'Cancel All': clears hijack state, defers consumer-side work.

        Audit + visual reset happen here; the process-side work (alert state
        reset, asking each handler to cancel) is delegated to the callback.
        """
        cleared = sorted(self.manual_override_streams)
        self.manual_override_streams.clear()
        self.audit.log("alert.cancel_all", cleared_hijacks=cleared)

        try:
            self._on_cancel_all()
        except Exception as e:
            logger.error(f"[{self.name}] on_cancel_all failed: {e}")

        # Reset the debug panel's per-stream visual indicators (if mounted).
        if self._debug_gui is not None:
            try:
                self._debug_gui.reset_all_hijacks()
            except Exception as e:
                logger.error(f"[{self.name}] reset_all_hijacks failed: {e}")

        logger.info(f"[{self.name}] CANCEL ALL invoked. Cleared {len(cleared)} hijack(s).")

    def _handle_terminate(self) -> None:
        """Operator 'Terminate Program': defers actual shutdown to the dispatcher."""
        self.audit.log("system.terminate_request", source="operator_gui")
        try:
            self._on_terminate()
        except Exception as e:
            logger.error(f"[{self.name}] on_terminate failed: {e}")
        logger.info(f"[{self.name}] Terminate requested from operator GUI.")
