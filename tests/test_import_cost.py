"""Importing a module must not load heavy libraries it does not need.

Measured on v2: the frame-grabber process imported torch only because its
logging helper lived next to `set_seed`, and the monitor process imported it
only to reach a result dataclass — ~500-600 MB RSS and 1.7-3 s of startup
each, for nothing. Every module that a torch-free process imports belongs in
`LIGHT_MODULES`; adding a heavy import to one of them fails here.

Each case runs in a fresh interpreter, since `sys.modules` in the test
process is already polluted by everything pytest has imported.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


HEAVY = ("cv2", "matplotlib", "numpy", "torch", "torchvision")

# module -> RSS budget in MB after importing it (bare interpreter: ~18 MB).
LIGHT_MODULES = {
    "birdcount": 25,
    "birdcount.core.log": 25,
    "birdcount.core.paths": 25,
    "birdcount.core.settings": 50,  # pydantic
}

_PROBE = """
import importlib, json, sys, psutil
importlib.import_module(sys.argv[1])
print(json.dumps({
    "heavy": sorted(m for m in sys.argv[2].split(",") if m in sys.modules),
    "rss_mb": psutil.Process().memory_info().rss / 2**20,
}))
"""


def _probe(module: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, module, ",".join(HEAVY)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", sorted(LIGHT_MODULES))
def test_light_module_stays_light(module: str) -> None:
    result = _probe(module)
    assert result["heavy"] == [], f"importing {module} loads {result['heavy']}"
    budget = LIGHT_MODULES[module]
    assert result["rss_mb"] <= budget, f"importing {module} costs {result['rss_mb']:.0f} MB RSS (budget {budget} MB)"
