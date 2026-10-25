"""Grabber and inference as real processes, talking through the frame ring."""

import multiprocessing as mp

import numpy as np
import pytest
import torch

from birdcount.core.model import DensityNet
from birdcount.deploy.grabber import GrabberProcess
from birdcount.deploy.inferencer import InferencerProcess
from birdcount.deploy.shm import FrameRing
from birdcount.deploy.sources import Source

from .conftest import make_layout, write_video


@pytest.mark.slow
def test_frames_flow_from_files_to_results(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    ckpt = tmp_path / "m.pth"
    torch.save(DensityNet().state_dict(), ckpt)
    sources = [Source(i, str(write_video(tmp_path / f"{i}.avi", 30)), is_file=True) for i in range(3)]
    ring = FrameRing.allocate(make_layout(streams=3), fps=5)
    results = ctx.Queue()
    grabber = GrabberProcess(ring.layout, sources, hw_decode=False)
    workers = [
        InferencerProcess(
            ring.layout,
            results,
            worker_id=w,
            num_workers=2,
            checkpoint=ckpt,
            device="cpu",
            mask_dir=None,
            macs=[None] * 3,
        )
        for w in range(2)
    ]
    try:
        grabber.start()
        for w in workers:
            w.start()
        seen = set()
        while len(seen) < 3:
            batch = results.get(timeout=60)
            ring.release(batch["sid"], batch["slot"])  # what the consumer does
            assert np.isfinite(batch["count"]).all() and (batch["inferred"] >= batch["captured"]).all()
            seen |= set(batch["sid"].tolist())
        assert {0, 2} <= {int(s) for s in seen}  # worker 0 serves cameras 0 and 2
    finally:
        for p in (grabber, *workers):
            p.stop()
        for p in (grabber, *workers):
            p.join(timeout=10)
        ring.close()
