"""The live runtime: processes, handlers, panels, and the loop that drives them.

    main process   App: this loop — results -> consumer -> handlers; Tk panels;
                   process supervision; camera watch. Never loads torch.
    Grabber        one decoding thread per camera -> frame ring
    Inferencer-N   frame ring -> counts + density -> result queue
    Monitor        on demand: frame ring -> camera grid window

Start-up: allocate the ring (its fixed name doubles as a single-instance
lock: a second runtime would drive the same devices and text the same
phones), start inference then the grabber, wait until every inference
worker has its model warm, then start the handlers and open the panels.

Shutdown — Terminate, Ctrl+C, closing the console, or a process that keeps
dying — always runs the same sequence: close the panels, stop the handlers
(devices off, last recording segment finished, evidence and logs flushed),
stop the processes, free the ring.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from birdcount.deploy.alarm import load_config
from birdcount.deploy.audit import AuditLog
from birdcount.deploy.consumer import Consumer
from birdcount.deploy.grabber import GrabberProcess
from birdcount.deploy.gui.controls import ControlState, Toggle
from birdcount.deploy.handlers.base import Handler
from birdcount.deploy.handlers.engine import AlarmEngine
from birdcount.deploy.inferencer import InferencerProcess
from birdcount.deploy.monitor.process import MonitorHandler
from birdcount.deploy.recorder import RecorderHandler
from birdcount.deploy.settings import DeploySettings
from birdcount.deploy.shm import FrameRing, RingLayout, SlotState
from birdcount.deploy.streams import Stream


logger = logging.getLogger(__name__)

RING_PREFIX = "birdcount"
QUEUE_PER_WORKER = 30  # ~3 s of batches at FPS 10: rides out a slow handler without dropping frames
RESULT_WAIT = 0.02  # s; the longest the loop waits for a batch, so the panels refresh at >= 50 Hz
SUPERVISE_EVERY = 0.5  # s
READY_TIMEOUT = 300.0  # s for every inference worker to load and warm up its model
MAX_RESTARTS = 5  # per process, then the whole runtime stops (non-zero exit: let a service restart it)
JOIN_TIMEOUT = 5.0


class AlreadyRunning(RuntimeError):
    pass


@dataclass
class Supervised:
    """A child process the main loop keeps alive."""

    factory: Callable[[], mp.Process]
    reclaim: Callable[[], int]  # frees ring slots the dead process held
    proc: mp.Process | None = None
    restarts: int = 0
    name: str = field(init=False, default="")

    def start(self) -> None:
        self.proc = self.factory()
        self.proc.start()
        self.name = self.proc.name

    def dead(self) -> bool:
        return self.proc is not None and not self.proc.is_alive()

    def restart(self) -> None:
        self.proc.join(timeout=0.5)
        freed = self.reclaim()
        self.restarts += 1
        logger.warning(
            "%s died (exit code %s); restart %d/%d%s.",
            self.name,
            self.proc.exitcode,
            self.restarts,
            MAX_RESTARTS,
            f", {freed} frame slot(s) reclaimed" if freed else "",
        )
        self.start()


class App:
    def __init__(self, settings: DeploySettings, streams: list[Stream], *, panels: bool = True):
        """`panels` False: never open the Tk panels (tests; Ctrl+C or `request_stop` ends the run)."""
        self.settings = settings
        self.streams = streams
        self.want_panels = panels
        self.macs = [s.mac for s in streams]
        self.exit_code = 0
        self._stop = threading.Event()
        self.stopped = threading.Event()  # set once shutdown has finished
        self._stop_reason = ""
        self.ring: FrameRing | None = None
        self.audit: AuditLog | None = None
        self.results: mp.Queue | None = None
        self.children: list[Supervised] = []
        self.inferencers: list[Supervised] = []
        self.handlers: list[Handler] = []
        self.consumer: Consumer | None = None
        self.panels = None

    # ------------------------------------------------------------------

    def request_stop(self, reason: str) -> None:
        if not self._stop.is_set():
            self._stop_reason = reason
            logger.info("Shutdown requested (%s).", reason)
            self._stop.set()

    def run(self) -> int:
        try:
            self._start()
            if not self._stop.is_set():
                self._loop()
        except AlreadyRunning:
            raise
        except Exception:
            logger.exception("Fatal error; shutting down.")
            self.exit_code = 1
        finally:
            self._shutdown()
            self.stopped.set()
        return self.exit_code

    # ------------------------------------------------------------------
    # Start-up
    # ------------------------------------------------------------------

    def _start(self) -> None:
        s = self.settings
        layout = RingLayout(RING_PREFIX, len(self.streams), s.num_buffers, s.frame_size)
        try:
            self.ring = FrameRing.allocate(layout, fps=s.fps)
        except FileExistsError:
            raise AlreadyRunning("another bird-count runtime is already running on this machine") from None
        self.ring.show_density, self.ring.show_mask = s.show_density_map, s.show_mask
        self.audit = AuditLog(s.audit_log_path)
        self.audit.log("system.start", streams=len(self.streams), source_type=s.source_type)

        workers = min(s.num_workers_per_gpu, len(self.streams))
        self.results = mp.Queue(maxsize=QUEUE_PER_WORKER * workers)
        self.inferencers = [
            Supervised(partial(self._inferencer, w, workers), partial(self._reclaim, w, workers, SlotState.READING))
            for w in range(workers)
        ]
        grabber = Supervised(self._grabber, partial(self._reclaim, 0, 1, SlotState.WRITING))
        self.children = [*self.inferencers, grabber]
        for child in self.children:  # inference first: it warms up while the cameras connect
            child.start()
        self._wait_until_ready()
        if self._stop.is_set():
            return

        self._build_handlers()
        for handler in self.handlers:
            handler.start()
        if self.want_panels:
            self._open_panels()
        logger.info("Running: %d camera(s), %d inference worker(s).", len(self.streams), workers)

    def _log_file(self, name: str):
        return self.settings.log_dir / f"deploy.{name.lower()}.log"

    def _inferencer(self, worker_id: int, workers: int) -> InferencerProcess:
        s = self.settings
        return InferencerProcess(
            self.ring.layout,
            self.results,
            worker_id=worker_id,
            num_workers=workers,
            checkpoint=s.model_path,
            device=f"cuda:{s.cuda_device}" if s.cuda_device.isdigit() else s.cuda_device,
            mask_dir=s.mask_dir,
            macs=self.macs,
            log_level=s.log_level,
            log_file=self._log_file(f"inferencer-{worker_id}"),
        )

    def _grabber(self) -> GrabberProcess:
        return GrabberProcess(
            self.ring.layout,
            [stream.source for stream in self.streams],
            hw_decode=self.settings.hw_decode,
            log_level=self.settings.log_level,
            log_file=self._log_file("grabber"),
        )

    def _reclaim(self, worker_id: int, workers: int, state: SlotState) -> int:
        sids = [sid for sid in range(len(self.streams)) if sid % workers == worker_id]
        return self.ring.reclaim(sids, state)

    def _wait_until_ready(self) -> None:
        logger.info("Waiting for %d inference worker(s) to load the model...", len(self.inferencers))
        deadline = time.monotonic() + READY_TIMEOUT
        while not all(child.proc.ready.is_set() for child in self.inferencers):
            self._supervise()  # a worker that dies while loading is restarted like any other
            if self._stop.wait(0.1):
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"inference not ready after {READY_TIMEOUT:.0f} s")

    def _build_handlers(self) -> None:
        s, audit, n = self.settings, self.audit, len(self.streams)
        config = load_config(s.alarm_config_path)
        simulate = s.debug and not (s.enable_sms_alarm and s.sms_alarm_real_worker)
        self.engine = AlarmEngine(config, self.macs, simulate_manual=simulate, audit=audit)
        self.control = ControlState(n, audit)

        self.monitor = MonitorHandler(
            self.ring,
            enabled=s.enable_monitor,
            audit=audit,
            fps=s.monitor_fps,
            labels=[f"CAM {sid}" for sid in range(n)],
            mask_dir=s.mask_dir,
            macs=self.macs,
            log_level=s.log_level,
            log_file=self._log_file("monitor"),
        )
        self.recorder = RecorderHandler(
            n,
            s.video_record_dir,
            fps=s.fps,
            frame_size=s.frame_size,
            segment_seconds=s.video_segment_seconds,
            fourcc=s.video_fourcc,
            enabled=s.enable_video_recorder,
            audit=audit,
        )
        handlers: list[Handler | None] = [self.monitor, self.recorder]
        if s.enable_smart_plug:
            from birdcount.deploy.handlers.plug import plug_handler

            plugs = {st.sid: st.zone.smart_plugs for st in self.streams if st.zone}
            handlers.append(
                plug_handler(
                    plugs,
                    self.engine.levels_for("smart_plug"),
                    email=s.tapo_email,
                    password=s.tapo_password.get_secret_value(),
                    audit=audit,
                )
            )
        if s.enable_speaker:
            from birdcount.deploy.handlers.speaker import speaker_handler

            speakers = {st.sid: st.zone.speakers for st in self.streams if st.zone}
            handlers.append(
                speaker_handler(
                    speakers,
                    self.engine.levels_for("speaker"),
                    username=s.speaker_username,
                    password=s.speaker_password.get_secret_value(),
                    clip=s.speaker_audio_file,
                    audit=audit,
                )
            )
        if s.enable_sms_alarm:
            from birdcount.deploy.handlers.sms import SmsHandler

            handlers.append(
                SmsHandler(
                    config,
                    output_dir=s.alarm_output_dir,
                    real_worker=s.sms_alarm_real_worker,
                    api_key=s.farm_sms_api_key.get_secret_value(),
                    audit=audit,
                )
            )
        if s.enable_count_logger:
            from birdcount.deploy.handlers.count_logger import CountLogger

            meta = {"frame_size": list(s.frame_size), "fps": s.fps, "mask_dir": str(s.mask_dir or "")}
            handlers.append(
                CountLogger(
                    s.count_log_dir,
                    s.count_log_hours,
                    self.engine,
                    self.macs,
                    meta=meta,
                    model_path=s.model_path,
                    audit=audit,
                )
            )
        self.handlers = [h for h in handlers if h is not None]
        logger.info("Handlers: %s.", ", ".join(h.name for h in self.handlers))
        self.consumer = Consumer(
            self.ring,
            self.engine,
            self.handlers,
            self.control,
            labels=[st.label for st in self.streams],
            audit=audit,
        )

    def _open_panels(self) -> None:
        try:
            from birdcount.deploy.gui.panels import Panels, debug_panel, operator_panel
            from birdcount.deploy.monitor.windows import screen_size
        except ImportError as e:  # a Python without Tk
            logger.warning("No GUI (%s); running headless. Stop with Ctrl+C.", e)
            return
        consumer, ring, recorder, monitor = self.consumer, self.ring, self.recorder, self.monitor
        operator = operator_panel(
            on_cancel_all=consumer.cancel_all,
            on_terminate=partial(self.request_stop, "operator"),
            monitor=Toggle("Monitor", monitor.is_enabled, monitor.set_enabled),
            active=consumer.active_devices,
            screen_width=screen_size()[0],
        )
        debug = None
        if self.settings.debug:
            debug = debug_panel(
                control=self.control,
                labels=[f"CAM {st.sid}\n{st.label}" for st in self.streams],
                devices_of={st.sid: set(st.devices) for st in self.streams},
                active_devices=lambda: set().union(*consumer.active_devices().values()),
                toggles=[
                    Toggle("Heatmap", lambda: ring.show_density, partial(monitor.set_overlay, "density")),
                    Toggle("Region masks", lambda: ring.show_mask, partial(monitor.set_overlay, "mask")),
                    Toggle("Record all", recorder.any_recording, recorder.set_all),
                ],
                recording=lambda sid: Toggle(
                    "REC", partial(recorder.is_recording, sid), partial(recorder.set_recording, sid)
                ),
                offline=consumer.offline,
            )
        panels = Panels(operator, debug)
        if panels.open():
            self.panels = panels
        else:
            logger.warning("No display for the panels; running headless. Stop with Ctrl+C.")

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        next_check = 0.0
        while not self._stop.is_set():
            try:
                records = self.results.get(timeout=RESULT_WAIT)
            except queue.Empty:
                records = None
            if records is not None:
                self.consumer.process(records)
            if self.panels is not None:
                self.panels.pump()
            now = time.monotonic()
            if now >= next_check:
                next_check = now + SUPERVISE_EVERY
                self._supervise()
                self.consumer.watch_cameras(now)

    def _supervise(self) -> None:
        for child in self.children:
            if not child.dead():
                continue
            if child.restarts >= MAX_RESTARTS:
                logger.critical("%s keeps dying (%d restarts); stopping the runtime.", child.name, child.restarts)
                self.audit.log("system.process_failed", process=child.name, restarts=child.restarts)
                self.exit_code = 1
                self.request_stop(f"{child.name} failed")
                return
            self.audit.log("system.process_restart", process=child.name, exit_code=child.proc.exitcode)
            child.restart()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("Shutting down%s...", f" ({self._stop_reason})" if self._stop_reason else "")
        self._stop.set()
        if self.panels is not None:
            _quietly(self.panels.close, "closing the panels")
        for handler in self.handlers:  # devices off, recordings finished, logs flushed
            _quietly(handler.stop, f"stopping {handler.name}")
        procs = [c.proc for c in self.children if c.proc is not None]
        for proc in procs:
            proc.stop()
        self._join(procs)
        if self.audit is not None:
            self.audit.log("system.stop", reason=self._stop_reason, exit_code=self.exit_code)
            self.audit.close()
        if self.ring is not None:
            self.ring.close()
        logger.info("Stopped.")

    def _join(self, procs: list[mp.Process]) -> None:
        """Wait for the children, draining the result queue: an inference worker
        cannot exit while its last batches are still waiting to be sent."""
        deadline = time.monotonic() + JOIN_TIMEOUT
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            try:
                while self.results is not None:
                    self.results.get_nowait()
            except queue.Empty:
                pass
            time.sleep(0.05)
        for proc in procs:
            if proc.is_alive():
                logger.warning("%s did not stop in %.0f s; terminating it.", proc.name, JOIN_TIMEOUT)
                proc.terminate()
                proc.join(timeout=2.0)


def _quietly(fn: Callable[[], None], what: str) -> None:
    try:
        fn()
    except Exception:
        logger.exception("Failed %s.", what)
