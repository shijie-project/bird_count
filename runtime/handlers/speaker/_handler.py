"""SpeakerHandler — broadcasts the deterrent clip on an alerting zone's speakers.

One `BroadcastLifecycle` per activated speaker, running on the handler's thread
pool. `handle()` only performs the atomic READY -> ACTIVE transition and
dispatches; everything network-facing lives in `_worker.py` / `_async.py`.

No SHM, no subprocess, no ack accounting — `handle()` returns as soon as the
lifecycle is submitted, so the consumer thread never blocks on the network.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from runtime.config import Config
from runtime.handlers import BaseHandler
from runtime.handlers._alarm_gate import AlarmDeviceGate, subscribed_levels
from runtime.inferencer import InferenceResult
from runtime.shared_memory import SharedMemoryConfig

from ._worker import BroadcastLifecycle, CancelSignal, httpx


logger = logging.getLogger(__name__)


# Device state machine values. Strings (vs IntEnum) keep audit logs readable.
_STATE_READY = "READY"
_STATE_ACTIVE = "ACTIVE"

# On stop() we signal every ACTIVE speaker and give the lifecycles this long to
# get their STOP request out before the process exits. Without the drain a
# shutdown mid-broadcast would leave the speakers playing.
_SHUTDOWN_DRAIN_TIMEOUT = 5.0
_SHUTDOWN_DRAIN_POLL = 0.1


class SpeakerHandler(BaseHandler):
    """Drives the zone speakers on alert.

    Trigger logic:
      - Starts on an alarm action whose level is in `actions.speaker.levels`
        (configs/alarm.json, Level 1 by default), or on a manual hijack, for
        the alarming stream's zone speakers — see `AlarmDeviceGate`.
      - Atomic READY -> ACTIVE transition per assigned speaker, then dispatches
        a `BroadcastLifecycle` on a ThreadPoolExecutor.
      - The speaker keeps broadcasting until every event holding it recovers
        (or its hijack is turned off), the operator stops it at the panel,
        `cancel_all()` fires from the GUI, or the safety cap expires.

    No aggregate on/off button: `config.envs.enable_speaker` is a config-time
    decision, and the handler is only constructed when it's set. The GUI surface
    is `cancel_all()` + `get_active_devices()` (both inherited slots on
    `BaseHandler`).
    """

    needs_frames = False  # this handler never reads raw video frames

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        name: str = "Speaker",
    ):
        super().__init__(config=config, shm_config=shm_config, name=name)

        # "Is the feature usable" (config flag AND httpx present) — not a GUI
        # toggle. cancel_all / get_active_devices early-out on it so the handler
        # stays safe when the wiring is missing.
        self._enabled = bool(config.envs.enable_speaker)
        if self._enabled and httpx is None:
            logger.error("[%s] 'httpx' library not installed. SpeakerHandler disabled.", self.name)
            self._enabled = False

        self.stream_to_speakers: dict[int, list[str]] = {
            sid: list(zone.speakers) for sid, zone in config.sid_to_zone.items()
        }
        self._all_unique_ips: list[str] = sorted({ip for ips in self.stream_to_speakers.values() for ip in ips})
        self.device_states: dict[str, str] = dict.fromkeys(self._all_unique_ips, _STATE_READY)
        self._gate = AlarmDeviceGate(self.stream_to_speakers, subscribed_levels(config, "speaker"))

        self._auth = (config.speaker_auth.username, config.speaker_auth.password)
        self._audio_file = config.envs.speaker_audio_file

        # Guards `device_states`: mutated from the consumer thread (handle) and
        # from pool threads (lifecycle teardown), read from the GUI thread.
        self._state_lock = threading.Lock()

        # One CancelSignal per IP, created upfront: `request()` is always safe
        # to call, bound to a loop or not, so there's no "lifecycle hasn't
        # registered yet" race for cancel_all to handle.
        self._cancel_signals: dict[str, CancelSignal] = {ip: CancelSignal() for ip in self._all_unique_ips}

        # Owned resource — created in start(), shut down in stop().
        self._executor: Optional[ThreadPoolExecutor] = None

        logger.info(
            "[%s] Initialized with %d unique device(s), on alarm level(s) %s.",
            self.name,
            len(self._all_unique_ips),
            sorted(self._gate.levels),
        )

    # ------------------------------------------------------------------
    # GUI surface  (consumed by ResultConsumer's cancel-all + status badges)
    # ------------------------------------------------------------------

    def cancel_all(self) -> None:
        """Signal every ACTIVE speaker to stop broadcasting."""
        if not self._enabled:
            return
        self._gate.clear()
        with self._state_lock:
            active_ips = [ip for ip, st in self.device_states.items() if st == _STATE_ACTIVE]
        for ip in active_ips:
            self._cancel_signals[ip].request()
        if active_ips:
            logger.info("[%s] cancel_all() signalled %d active speaker(s): %s", self.name, len(active_ips), active_ips)
            self.audit.log("handler.cancel_all", handler=self.name, devices=active_ips)

    def get_active_devices(self) -> set[str]:
        if not self._enabled:
            return set()
        with self._state_lock:
            return {ip for ip, st in self.device_states.items() if st == _STATE_ACTIVE}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()
        if not self._enabled:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(self.device_states)),
            thread_name_prefix=self.name,
        )
        self.audit.log("speaker.devices", devices=sorted(self.device_states))

    def stop(self) -> None:
        if self._enabled:
            self.cancel_all()
            self._drain_active()
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None
            logger.info("[%s] Handler stopped.", self.name)
        super().stop()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]) -> None:
        if not self._enabled or self._executor is None:
            return

        switch_on, switch_off = self._gate.update(result)
        for ip in switch_on:
            # Atomic READY -> ACTIVE; skip if this speaker is already broadcasting.
            with self._state_lock:
                if self.device_states.get(ip) != _STATE_READY:
                    continue
                self.device_states[ip] = _STATE_ACTIVE
            self.audit.log(
                "device.activate",
                handler=self.name,
                ip=ip,
                stream_id=result.stream_id,
                count=result.count,
                level=result.alarm_level,
                manual=result.manual_override,
            )
            self._executor.submit(self._run_lifecycle, ip)

        for ip in switch_off:
            # Only an ACTIVE speaker has a lifecycle to stop. Cancelling a READY
            # one would pre-arm its signal and kill the *next* broadcast at birth.
            with self._state_lock:
                if self.device_states.get(ip) != _STATE_ACTIVE:
                    continue
                self._cancel_signals[ip].request()
            self.audit.log("device.release", handler=self.name, ip=ip, stream_id=result.stream_id)

    # ------------------------------------------------------------------
    # Thread-pool plumbing
    # ------------------------------------------------------------------

    def _run_lifecycle(self, ip: str) -> None:
        """ThreadPool entry point: run one broadcast, then return to READY."""
        signal = self._cancel_signals[ip]
        lifecycle = BroadcastLifecycle(
            ip=ip,
            auth=self._auth,
            audio_file=self._audio_file,
            cancel_signal=signal,
            tag=self.name,
        )
        try:
            error_msg = lifecycle.run()
        finally:
            with self._state_lock:
                self.device_states[ip] = _STATE_READY
                # Re-unbind under the state lock so a cancel that landed between
                # the lifecycle's own unbind and now is discarded — otherwise
                # it'd be carried into the next activation as a stale pre-set.
                signal.unbind()
        self.audit.log("device.deactivate", handler=self.name, ip=ip, error=error_msg)
        logger.debug("[%s] %s reset to READY state.", self.name, ip)

    def _drain_active(self) -> None:
        """Wait (briefly) for cancelled lifecycles to finish issuing STOP."""
        deadline = time.monotonic() + _SHUTDOWN_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            active = self.get_active_devices()
            if not active:
                return
            time.sleep(_SHUTDOWN_DRAIN_POLL)
        remaining = self.get_active_devices()
        if remaining:
            logger.warning(
                "[%s] %d speaker(s) still ACTIVE after %.1fs drain; they may keep playing: %s",
                self.name,
                len(remaining),
                _SHUTDOWN_DRAIN_TIMEOUT,
                sorted(remaining),
            )
