"""Importing a module must not load heavy libraries it does not need.

Processes pay for every import of their entry module, once at startup and
then as RSS for their whole lifetime. Each module below declares what it may
not load and an RSS budget; adding a heavy import to one of them fails here.

Each case runs in a fresh interpreter, since `sys.modules` in the test
process is already polluted by everything pytest has imported.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


_NO_HEAVY = ("cv2", "matplotlib", "numpy", "torch", "torchvision")
_NO_TORCH = ("matplotlib", "torch", "torchvision")

# module -> (modules it must not load, RSS budget in MB after importing it).
# Bare interpreter: ~18 MB; torch alone: ~510 MB.
CASES = {
    "birdcount": (_NO_HEAVY, 25),
    "birdcount.core.log": (_NO_HEAVY, 25),
    "birdcount.core.paths": (_NO_HEAVY, 25),
    "birdcount.core.settings": (_NO_HEAVY, 50),  # pydantic
    "birdcount.core.camera_identity": (_NO_HEAVY, 25),
    # Frame / mask / drawing code runs in torch-free processes (grabber, monitor).
    "birdcount.core.frames": (_NO_TORCH, 60),
    "birdcount.core.masks": (_NO_TORCH, 60),
    "birdcount.core.visualize": (_NO_TORCH, 60),
    # torch is the point of these; torchvision would add ~1 s and ~100 MB.
    "birdcount.core.model": (("cv2", "matplotlib", "torchvision"), 560),
    "birdcount.core.inference": (("matplotlib", "torchvision"), 560),
}

_PROBE = """
import importlib, json, sys, psutil
importlib.import_module(sys.argv[1])
print(json.dumps({
    "loaded": sorted(m for m in sys.argv[2].split(",") if m in sys.modules),
    "rss_mb": psutil.Process().memory_info().rss / 2**20,
}))
"""


@pytest.mark.parametrize("module", sorted(CASES))
def test_import_stays_light(module: str) -> None:
    forbidden, budget = CASES[module]
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, module, ",".join(forbidden)],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["loaded"] == [], f"importing {module} loads {result['loaded']}"
    assert result["rss_mb"] <= budget, f"importing {module} costs {result['rss_mb']:.0f} MB RSS (budget {budget} MB)"
