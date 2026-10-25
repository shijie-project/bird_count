"""What a run writes to disk, written atomically.

    <run>/last.tar                  everything needed to resume (written at each validation and at the end)
    <run>/best_ep0660_mae3.77_rmse5.12.pth
                                    validated (EMA) weights, plain state dict: what deployment loads
                                    (v2 called the RMSE "mse" in these names)

Every file is written to a temporary name and renamed into place, so a crash
mid-write leaves the previous file intact instead of a truncated one.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import torch


logger = logging.getLogger(__name__)

BEST_RE = re.compile(r"best_ep(?P<epoch>\d+)_mae(?P<mae>[\d.]+)_rmse(?P<rmse>[\d.]+)\.pth$")


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def best_name(epoch: int, mae: float, rmse: float) -> str:
    return f"best_ep{epoch:04d}_mae{mae:.2f}_rmse{rmse:.2f}.pth"


class BestCheckpoints:
    """The `keep` most recent best checkpoints; older ones are deleted.

    Picks up best files already in the folder, so a resumed run keeps rotating
    them instead of leaking the earlier ones.
    """

    def __init__(self, run_dir: Path, keep: int):
        self.run_dir = run_dir
        self.keep = keep
        found = [(int(m["epoch"]), p) for p in run_dir.glob("best_ep*.pth") if (m := BEST_RE.match(p.name))]
        self.paths = [p for _, p in sorted(found)]

    def add(self, state: dict, epoch: int, mae: float, rmse: float) -> Path:
        path = self.run_dir / best_name(epoch, mae, rmse)
        save_atomic(state, path)
        self.paths.append(path)
        while len(self.paths) > self.keep:
            old = self.paths.pop(0)
            try:
                old.unlink()
            except OSError as e:
                logger.warning("Could not delete %s: %s", old.name, e)
        return path

    @property
    def latest(self) -> Path | None:
        return self.paths[-1] if self.paths else None
