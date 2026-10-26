"""Run the live runtime.

    python -m birdcount.deploy            # run (also: birdcount-deploy)
    python -m birdcount.deploy --check    # validate the configuration, list the streams, exit

`--check` loads everything a run needs — settings, topology, streams, alarm
config, checkpoint, masks — and prints what each camera will get (zone,
devices, region mask, alarm threshold), without starting anything. Exit code
0 when the configuration is usable, 1 when it is not.

Exit codes of a run: 0 stopped normally, 1 failed (a process kept dying, or
a fatal error), 2 not started (configuration error, or already running).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys

from birdcount.core.camera_identity import index_by_mac
from birdcount.core.frames import IMAGE_SUFFIXES
from birdcount.core.log import setup_logging
from birdcount.deploy.alarm import load_config
from birdcount.deploy.alarm.camera_ids import resolve_camera_id
from birdcount.deploy.settings import DeploySettings
from birdcount.deploy.streams import Stream, StreamError, describe, resolve_streams
from birdcount.deploy.topology import TopologyError, load_topology


logger = logging.getLogger("birdcount.deploy")


def _prepare(settings: DeploySettings) -> list[Stream]:
    """Everything that can be validated before a process starts; raises on the first error."""
    if settings.model_path is None or not settings.model_path.is_file():
        raise FileNotFoundError(f"MODEL_PATH {settings.model_path or '(blank)'} is not a file")
    try:
        load_config(settings.alarm_config_path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"ALARM_CONFIG_PATH {settings.alarm_config_path}: {e}") from e
    topology = load_topology(settings.topology_path)
    streams = resolve_streams(settings, topology)
    logger.info("%d stream(s) from %s:", len(streams), settings.source_type)
    for line in describe(streams):
        logger.info("  %s", line)
    return streams


def _check(settings: DeploySettings, streams: list[Stream]) -> None:
    config = load_config(settings.alarm_config_path)
    masks = {}
    if settings.mask_dir is not None and settings.mask_dir.is_dir():
        masks, _ = index_by_mac(p for p in settings.mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"\n{'sid':<4} {'stream':<24} {'MAC':<13} {'zone':<22} {'devices':>7} {'threshold':>9}  mask")
    incomplete = 0
    for s in streams:
        camera_id = resolve_camera_id(mac=s.mac, known=config.cameras.keys())
        threshold = config.cameras[camera_id].threshold if camera_id else None
        mask = masks.get(s.mac) if s.mac else None
        flag = "" if threshold is not None and mask else "  <- incomplete"
        incomplete += bool(flag)
        print(
            f"{s.sid:<4} {s.label[:24]:<24} {s.mac or '-':<13} {(s.zone.name if s.zone else '-')[:22]:<22} "
            f"{len(s.devices):>7} {f'{threshold:.0f}' if threshold is not None else '-':>9}  "
            f"{mask.name if mask else '-'}{flag}"
        )
    print(
        f"\n{len(streams)} stream(s); {incomplete} without both a threshold and a mask "
        "(no threshold: never alarms; no mask: counts the whole frame)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m birdcount.deploy", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="validate the configuration and exit")
    args = parser.parse_args(argv)

    settings = DeploySettings.load()
    setup_logging(settings.log_level, log_file=None if args.check else settings.log_dir / "deploy.log")
    try:
        streams = _prepare(settings)
        if args.check:
            _check(settings, streams)
            return 0
    except (OSError, TopologyError, StreamError, ValueError) as e:
        logger.error("Configuration error: %s", e)
        return 1 if args.check else 2

    from birdcount.deploy.app import AlreadyRunning, App
    from birdcount.deploy.lifecycle import on_shutdown_request

    app = App(settings, streams)
    on_shutdown_request(app.request_stop, app.stopped)
    try:
        return app.run()
    except AlreadyRunning as e:
        logger.error("Not started: %s.", e)
        return 2


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
