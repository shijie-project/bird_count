"""`CancelSignal` — the cancel primitive shared by every device handler.

Device handlers (smart plug, speaker) run one asyncio lifecycle per device on
a thread pool, and the GUI's "Cancel All" button has to reach into those loops
from the consumer thread. This is the bridge: `request()` is plain-threadsafe,
`wait()` is native asyncio.

Lives at the handler-package root because both device handlers need exactly the
same semantics — including the pre-bind buffering that removes the "lifecycle
hasn't registered yet" race.
"""

import asyncio
import threading
from typing import Optional


class CancelSignal:
    """Pre-creatable cancel signal that bridges sync (cross-thread) `request()`
    with native `asyncio.Event.wait()` inside a loop.

    Designed so a handler can create one *per device at init time* — before any
    asyncio loop exists — and still have `request()` work safely from the GUI
    thread, with or without a loop currently bound. This eliminates the
    registration-window race that an in-loop-only signal forces you to handle.

    Lifecycle:
        signal = CancelSignal()              # handler init, no loop needed
        signal.request()                     # safe — sets _pre_requested flag
        ...
        signal.bind()                        # inside the asyncio loop (lifecycle entry)
        # — any pre-bind request() carried forward by setting the event
        await signal.wait(timeout=...)       # native asyncio wait, no executor thread
        ...
        signal.unbind()                      # lifecycle exit; clears state for reuse
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event: Optional[asyncio.Event] = None
        self._pre_requested = False

    def request(self) -> None:
        """Thread-safe — fire cancel from any thread, with or without bind().

        If no loop is bound, flips an internal flag; the next `bind()` will
        immediately set the freshly created event so the first `wait()`
        returns `True`.
        """
        with self._lock:
            if self._event is None:
                self._pre_requested = True
                return
            loop, event = self._loop, self._event
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            # Loop closed between our unlock and the schedule call — the
            # lifecycle is already tearing down, so the cancel is moot.
            pass

    def bind(self) -> None:
        """Bind to the running loop and create the underlying `asyncio.Event`.

        Must be called from inside the loop that will `await wait()`. If a
        `request()` arrived before binding, the new event is set immediately.
        """
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._event = asyncio.Event()
            if self._pre_requested:
                self._event.set()
                self._pre_requested = False

    def unbind(self) -> None:
        """Reset to the pre-bind state. Idempotent — safe to call twice.

        Also discards any cancel that arrived after the loop closed; the
        next `bind()` starts on a clean slate.
        """
        with self._lock:
            self._loop = None
            self._event = None
            self._pre_requested = False

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Returns True if the event was set within `timeout`, False on timeout.

        Must be called after `bind()`. `timeout=None` waits forever.
        """
        if self._event is None:
            raise RuntimeError("CancelSignal.wait() called before bind()")
        if timeout is None:
            await self._event.wait()
            return True
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
