"""Starting and stopping the runtime's processes cleanly, however it ends.

Children (grabber, inference, monitor) call `init_child` first thing in
`run()`:

* Ctrl+C reaches every process attached to the console. Children ignore it
  and are stopped by the main process in order, instead of each dying on its
  own KeyboardInterrupt mid-write.
* A child whose parent is gone stops itself within about a second. v2's
  children outlived a killed main process, keeping the GPU and every camera
  connection until someone found them in Task Manager.

The main process calls `on_shutdown_request` once: Ctrl+C, Ctrl+Break and
closing the console window all become the same orderly shutdown as the
operator's Terminate button. Closing the window used to kill the process on
the spot, leaving a plug switched on and the current recording unfinished.
Windows gives a closing console ~5 s; the handler holds the close until the
shutdown has finished or that budget is spent.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from birdcount.core.log import setup_logging


logger = logging.getLogger(__name__)

# s an orphan gets to stop by itself before it exits hard. Short: with the
# parent gone there is nobody to hand anything to, and a worker blocked on
# the dead parent's result pipe would never finish on its own.
ORPHAN_GRACE = 1.0
CLOSE_BUDGET = 4.5  # s of Windows' ~5 s console-close allowance spent shutting down

_CTRL_C, _CTRL_BREAK, _CTRL_CLOSE, _CTRL_LOGOFF, _CTRL_SHUTDOWN = 0, 1, 2, 5, 6
_console_handler = None  # the ctypes callback must stay referenced while installed


def init_child(log_level: str, log_file: Path | None, stop_event) -> None:
    """Logging, ignore console interrupts, and stop when the parent dies."""
    setup_logging(log_level, log_file=log_file)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    parent = mp.parent_process()
    if parent is not None:
        threading.Thread(target=_watch_parent, args=(parent, stop_event), name="ParentWatch", daemon=True).start()


def _watch_parent(parent, stop_event) -> None:
    parent.join()  # returns when the parent process is gone
    if stop_event.is_set():
        return  # already stopping: the parent exited normally after asking us to
    logger.error("Parent process is gone; stopping.")
    stop_event.set()
    threading.Event().wait(ORPHAN_GRACE)
    os._exit(1)


def on_shutdown_request(request: Callable[[str], None], finished: threading.Event) -> None:
    """Route Ctrl+C / Ctrl+Break / console close to `request(reason)`.

    `finished` is set by the caller once shutdown is complete; a console close
    waits for it (within Windows' allowance) before letting the process die.
    """
    signal.signal(signal.SIGINT, lambda *_: request("ctrl+c"))
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: request("ctrl+break"))
    if sys.platform == "win32":
        _install_console_close(request, finished)


def _install_console_close(request: Callable[[str], None], finished: threading.Event) -> None:
    global _console_handler
    import ctypes
    from ctypes import wintypes

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    def handler(event: int) -> bool:
        if event not in (_CTRL_CLOSE, _CTRL_LOGOFF, _CTRL_SHUTDOWN):
            return False  # Ctrl+C / Break: let the signal handlers above take it
        request("console closed")
        finished.wait(CLOSE_BUDGET)
        return True

    if ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True):
        _console_handler = handler
    else:
        logger.warning("Could not hook console close; closing the window will skip the clean shutdown.")
