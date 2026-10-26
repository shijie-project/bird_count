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
    # The grabber, the monitor and the main process never load torch; the
    # inferencer module defers torch to its process's run().
    "birdcount.deploy.shm": (_NO_TORCH, 60),
    "birdcount.deploy.results": (_NO_TORCH, 60),
    "birdcount.deploy.grabber": (_NO_TORCH, 60),
    "birdcount.deploy.inferencer": (_NO_TORCH, 60),
    # The alarm core is stdlib only; handlers run on the torch-free main process,
    # and device libraries load only when a device is driven.
    "birdcount.deploy.alarm": (_NO_HEAVY, 35),
    "birdcount.deploy.handlers.sms": ((*_NO_TORCH, "httpx", "tapo"), 60),
    "birdcount.deploy.handlers.speaker": ((*_NO_TORCH, "httpx", "tapo"), 60),
    "birdcount.deploy.handlers.plug": ((*_NO_TORCH, "httpx", "tapo"), 60),
    "birdcount.deploy.handlers.count_logger": ((*_NO_TORCH, "httpx", "tapo"), 70),
    # The monitor process and the recorder threads draw and encode with cv2 only;
    # the panels are Tk on the main process.
    "birdcount.deploy.monitor.process": ((*_NO_TORCH, "httpx", "tapo"), 60),
    "birdcount.deploy.recorder": ((*_NO_TORCH, "httpx", "tapo"), 60),
    "birdcount.deploy.gui.panels": ((*_NO_HEAVY, "httpx", "tapo"), 35),
    # The main process: no torch, and no device libraries or Tk until used.
    "birdcount.deploy.__main__": ((*_NO_TORCH, "httpx", "tapo", "tkinter"), 75),
    "birdcount.deploy.app": ((*_NO_TORCH, "httpx", "tapo", "tkinter"), 75),
    "birdcount.deploy.consumer": ((*_NO_TORCH, "httpx", "tapo"), 50),
    "birdcount.deploy.topology": (_NO_HEAVY, 30),
    "birdcount.deploy.lifecycle": (_NO_HEAVY, 25),
    "birdcount.deploy.tools.devices": ((*_NO_HEAVY, "httpx", "tapo"), 45),
    # Training data runs in every DataLoader worker: no torchvision, no PIL.
    "birdcount.training.data.loader": (("matplotlib", "torchvision", "PIL"), 580),
    "birdcount.training.data.photometric": (("matplotlib", "torchvision", "PIL"), 560),
    # matplotlib loads only when a timeline figure is drawn.
    "birdcount.training.video.__main__": (("matplotlib", "torchvision"), 580),
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
