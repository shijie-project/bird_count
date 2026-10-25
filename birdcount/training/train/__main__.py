"""Train the density model.

    python -m birdcount.training.train                       # the production recipe
    python -m birdcount.training.train --lr 5e-5 --name lr5e-5
    python -m birdcount.training.train --resume outputs/train/20261026-101500

Writes a run folder (OUTPUT_ROOT/train/<time>[-name]): config.json,
train.log, metrics.jsonl, summary.json, last.tar, best_*.pth. Ctrl+C stops
after saving the state of the last finished epoch; `--resume` continues it.
"""

from __future__ import annotations

import sys

from birdcount.core.settings import CoreSettings
from birdcount.training.train.config import build_parser, config_from_args
from birdcount.training.train.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume is not None:
        changed = sorted(k for k, v in vars(args).items() if k != "resume" and v != parser.get_default(k))
        if changed:
            parser.error(
                f"--resume continues with the run's own config; drop {', '.join('--' + c.replace('_', '-') for c in changed)}"
            )
    trainer = Trainer(config_from_args(args), CoreSettings.load())
    trainer.setup()
    summary = trainer.run()
    return 0 if summary["status"] != "interrupted" else 130


if __name__ == "__main__":
    sys.exit(main())
