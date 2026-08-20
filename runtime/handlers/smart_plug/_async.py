import asyncio
import logging
import math
import time
from typing import Optional

from runtime.handlers._cancel import CancelSignal


try:
    from tapo import ApiClient
except ImportError:  # pragma: no cover - depends on deployment
    ApiClient = None


logger = logging.getLogger(__name__)


# OFF is safety-critical — retry once on transient failure before giving up.
_OFF_RETRY_ATTEMPTS = 2


# Re-exported: `_worker.py` imports CancelSignal from here alongside the
# lifecycle coroutine. The class itself now lives in `runtime.handlers._cancel`
# because the speaker handler needs the identical primitive.
__all__ = ["ApiClient", "CancelSignal", "run_plug_lifecycle"]


async def _turn_off(
    device,
    *,
    ip: str,
    reason: str,
    on_off_timeout: float,
    tag: str,
) -> None:
    """Send OFF with retry. Raises on final failure so the caller is notified
    that the plug may still be live."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, _OFF_RETRY_ATTEMPTS + 1):
        try:
            await asyncio.wait_for(device.off(), timeout=on_off_timeout)
            logger.info(f"[{tag}] {ip} OFF issued ({reason}).")
            return
        except Exception as e:
            last_exc = e
            if attempt < _OFF_RETRY_ATTEMPTS:
                logger.warning(f"[{tag}] OFF attempt {attempt} failed on {ip} ({reason}), retrying: {e}")
            else:
                logger.error(f"[{tag}] OFF failed after {attempt} attempts on {ip} ({reason}): {e}")
    assert last_exc is not None
    raise last_exc


async def run_plug_lifecycle(
    *,
    ip: str,
    api_client: "ApiClient",
    tag: str,
    cancel_signal: CancelSignal,
    max_idle_time: Optional[float] = None,
    on_off_timeout: float,
    poll_timeout: float,
    poll_interval: float,
    max_consecutive_poll_failures: int,
) -> None:
    """One-shot async lifecycle for a single Tapo P100 plug.

    The caller passes a shared `api_client` (one per handler — the tapo Rust
    backend manages its own tokio runtime, so the same instance is safe to
    use from multiple asyncio loops / threadpool workers concurrently) and a
    pre-created `cancel_signal`. We bind the signal to the running loop on
    entry and unbind on exit; any `request()` that arrives before bind() is
    carried forward automatically by the signal.

    Flow:
      1. Turn the plug ON.
      2. Loop until one of:
           - `cancel_signal` is set (GUI cancel_all) → issue explicit OFF.
           - `device_on` reads False (operator manually killed it) → exit.
           - `max_idle_time` elapses (safety timeout) → issue explicit OFF.
      3. `cancel_signal.unbind()` in `finally`.
    """
    if ApiClient is None or api_client is None:
        return

    cancel_signal.bind()
    try:
        device = await api_client.p100(ip)
        await asyncio.wait_for(device.on(), timeout=on_off_timeout)
        logger.info(f"[{tag}] {ip} is now ON.")

        # Use monotonic clock for the deadline so NTP adjustments or DST
        # changes can't extend/shorten the safety window. `max_idle_time=None`
        # means "no timeout — only cancel_all() or a manual operator shutoff
        # can end the lifecycle"; we encode that as an infinite deadline so
        # the loop condition stays a simple comparison.
        deadline = math.inf if max_idle_time is None else time.monotonic() + max_idle_time
        consecutive_failures = 0

        while time.monotonic() < deadline:
            cancelled = await cancel_signal.wait(timeout=poll_interval)
            if cancelled:
                logger.info(f"[{tag}] Cancel signal for {ip}. Turning OFF.")
                await _turn_off(device, ip=ip, reason="cancel", on_off_timeout=on_off_timeout, tag=tag)
                return

            try:
                info = await asyncio.wait_for(device.get_device_info(), timeout=poll_timeout)
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_poll_failures:
                    logger.error(f"[{tag}] {consecutive_failures} consecutive poll failures on {ip}, giving up: {e}")
                    try:
                        await _turn_off(
                            device, ip=ip, reason="poll-failure-giveup", on_off_timeout=on_off_timeout, tag=tag
                        )
                    except Exception:
                        pass  # already logged inside _turn_off
                    raise
                continue

            consecutive_failures = 0
            if not info.device_on:
                logger.info(f"[{tag}] Manual shutdown detected on {ip}. Done.")
                return

        logger.warning(f"[{tag}] Max idle time ({max_idle_time}s) reached for {ip}. Turning OFF.")
        await _turn_off(device, ip=ip, reason="max-idle-timeout", on_off_timeout=on_off_timeout, tag=tag)
    finally:
        cancel_signal.unbind()
