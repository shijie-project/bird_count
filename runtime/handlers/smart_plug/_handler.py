import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import numpy as np

from runtime.config import Config
from runtime.handlers import BaseHandler, BaseHandlerGUIMixin
from runtime.inference_process import InferenceResult
from runtime.shared_memory import SharedMemoryConfig

from ._worker import ApiClient, CancelSignal, PlugLifecycle


logger = logging.getLogger(__name__)


# Device state machine values. Strings (vs IntEnum) keep audit logs human-readable.
_STATE_READY = "READY"
_STATE_ACTIVE = "ACTIVE"


class SmartPlugGUIMixin(BaseHandlerGUIMixin):
    """
    GUI-callable surface for SmartPlugHandler.

    The plug feature has no aggregate on/off button (the GUI never disables
    the handler at runtime — that's a config-time decision). The GUI only
    needs:

        - cancel_all()        : abort every ACTIVE plug.
        - get_active_devices(): which IPs are currently ON, for status badges.

    All state lives on SmartPlugHandler; the mixin reads it through the
    handler's lock so the GUI thread can't observe a torn READY<->ACTIVE
    transition.
    """

    # Attributes / methods provided by SmartPlugHandler — declared here so
    # the mixin's intent is readable in isolation.
    name: str
    _enabled: bool
    device_states: dict[str, str]
    _cancel_signals: dict[str, CancelSignal]

    @property
    def state_lock(self) -> threading.Lock: ...

    def cancel_all(self):
        """Signal every ACTIVE plug to turn OFF.

        `CancelSignal.request()` is always safe to call — if the lifecycle
        hasn't yet bound the signal to its loop, the request is buffered and
        applied on bind. No registration race to handle here.
        """
        if not self._enabled:
            return
        with self.state_lock:
            active_ips = [ip for ip, st in self.device_states.items() if st == _STATE_ACTIVE]
        for ip in active_ips:
            self._cancel_signals[ip].request()
        if active_ips:
            logger.info(f"[{self.name}] cancel_all() signalled {len(active_ips)} active plug(s): {active_ips}")
            if self.audit:
                self.audit.log("handler.cancel_all", handler=self.name, devices=active_ips)

    def get_active_devices(self) -> set[str]:
        if not self._enabled:
            return set()
        with self.state_lock:
            return {ip for ip, st in self.device_states.items() if st == _STATE_ACTIVE}


class SmartPlugHandler(SmartPlugGUIMixin, BaseHandler):
    """
    Industrial Handler for IoT Plugs (TP-Link Tapo).

    Trigger Logic:
      - Watches `alert_flag` on each InferenceResult.
      - Atomic READY -> ACTIVE transition per assigned plug, then dispatches
        a PlugLifecycle on a ThreadPoolExecutor.
      - The plug stays in ACTIVE until either the operator turns it off
        manually OR cancel_all() is invoked from the GUI.
    """

    needs_frames = False  # this handler never reads raw video frames

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        name: str = "SmartPlug",
        max_idle_time: Optional[Union[float, int]] = None,
    ):
        super().__init__(config=config, shm_config=shm_config, name=name)

        # _enabled reflects "is the feature usable" (config flag AND library
        # present). It is NOT a GUI-toggle — there's no enable/disable UI for
        # smart plugs. The mixin's cancel_all/get_active_devices early-out on
        # this flag so the handler stays safe when wiring is missing.
        self._enabled = bool(config.envs.enable_smart_plug)
        if self._enabled and ApiClient is None:
            logger.error(f"[{self.name}] 'tapo' library not installed. SmartPlugHandler disabled.")
            self._enabled = False
        if not self._enabled:
            return

        self.max_idle_time = max_idle_time
        self.auth_email = config.plug_auth.email
        self.auth_password = config.plug_auth.password

        self.stream_to_plugs: dict[int, set[str]] = {sid: zone.smart_plugs for sid, zone in config.sid_to_zone.items()}
        self._all_unique_ips: list[str] = sorted({ip for ips in self.stream_to_plugs.values() for ip in ips})
        self.device_states: dict[str, str] = dict.fromkeys(self._all_unique_ips, _STATE_READY)

        # One CancelSignal per IP, created upfront in start() (after spawn —
        # signals contain a threading.Lock which can't be pickled). Pre-creating
        # them eliminates the "lifecycle hasn't registered yet" race that an
        # in-loop-only signal would force on us: `signal.request()` is always
        # safe to call, with or without a loop bound.
        self._cancel_signals: dict[str, CancelSignal] = {}

        # threading.Lock and ThreadPoolExecutor still can't be pickled across
        # spawn — lazy-create in the child process.
        self._state_lock: Optional[threading.Lock] = None
        self._executor: Optional[ThreadPoolExecutor] = None

        # Single shared ApiClient for all lifecycles. Created in start() (after
        # spawn) because the tapo Rust binding wraps a tokio runtime and we'd
        # rather not pickle that across processes. Thread-safe by design — the
        # Rust side serialises access to its internal state — so all of the
        # executor's worker threads can call `api_client.p100(...)` concurrently
        # against the same instance.
        self._api_client: Optional[ApiClient] = None

        logger.info(f"[{self.name}] Initialized with {len(self._all_unique_ips)} unique devices.")

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            num_threads = max(1, len(self.device_states))
            self._executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix=self.name)
        return self._executor

    @property
    def state_lock(self) -> threading.Lock:
        if self._state_lock is None:
            self._state_lock = threading.Lock()
        return self._state_lock

    def start(self):
        super().start()
        if self._enabled:
            # Eagerly construct the shared client and the per-IP cancel signals
            # now that we're in the child process (both contain unpicklable
            # state — Rust tokio runtime / threading.Lock).
            self._api_client = ApiClient(self.auth_email, self.auth_password)
            self._cancel_signals = {ip: CancelSignal() for ip in self._all_unique_ips}
            if self.audit:
                self.audit.log("smart_plug.devices", devices=sorted(self.device_states.keys()))

    def stop(self):
        if self._enabled:
            self.cancel_all()
            self.executor.shutdown(wait=False)
            logger.info(f"[{self.name}] Handler stopped.")
        super().stop()

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]):
        if not self._enabled or not result.alert_flag:
            return

        target_ips = self.stream_to_plugs.get(result.stream_id, [])
        for ip in target_ips:
            # Atomic READY -> ACTIVE; skip if already ACTIVE on this device.
            with self.state_lock:
                if self.device_states.get(ip) != _STATE_READY:
                    continue
                self.device_states[ip] = _STATE_ACTIVE
            if self.audit:
                self.audit.log(
                    "device.activate",
                    handler=self.name,
                    ip=ip,
                    stream_id=result.stream_id,
                    count=result.count,
                )
            self.executor.submit(self._run_lifecycle, ip)

    def _run_lifecycle(self, ip: str):
        """ThreadPool entry point: run one PlugLifecycle, then return to READY."""
        assert self._api_client is not None, "start() must run before _run_lifecycle"
        signal = self._cancel_signals[ip]
        lifecycle = PlugLifecycle(
            ip=ip,
            api_client=self._api_client,
            cancel_signal=signal,
            max_idle_time=self.max_idle_time,
            tag=self.name,
        )
        try:
            error_msg = lifecycle.run()
        finally:
            with self.state_lock:
                self.device_states[ip] = _STATE_READY
                # Re-unbind under state_lock so any cancel that fired between
                # the lifecycle's own unbind and now is discarded — otherwise
                # it'd be carried into the next activation as a stale pre-set.
                signal.unbind()
        if self.audit:
            self.audit.log("device.deactivate", handler=self.name, ip=ip, error=error_msg)
        logger.debug(f"[{self.name}] {ip} reset to READY state.")
