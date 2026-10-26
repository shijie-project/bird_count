"""The one fixed standard a picture is judged by: five levels of density.

Every rating, by a model or compared against a person's label, uses this
text unchanged, so ratings from different days and runs mean the same thing.
Changing it starts a new standard: bump RUBRIC_VERSION, which is stored with
every rating, and old ratings stop counting for new proposals.
"""

from __future__ import annotations

import json
import re


# 2: a full shed is packed as a rule, so only piled-up birds (5) count as a
# pile-up, and 4 / 5 are told apart by whether birds are on top of others.
# 3: the area outside the mask shaded darker (SHADE 0.3 -> 0.15).
# 4: density only: whether birds pile up is the count threshold's call, not the
# model's. Five steps of floor cover, the top two split where a full shed sits;
# the whole bright area is judged, and the birds are always chickens (a bluish
# camera's pictures were taken for fish in a tank).
RUBRIC_VERSION = 4
DENSE_LEVEL = 5  # a rating at or above this is crowded enough for the alarm
# Brightness kept outside the camera's mask, in every picture judged by the rubric: the
# model's (calibrate.images) and the person's (the farm backend's labelling page).
SHADE = 0.15

LEVELS = {
    1: "Sparse: most of the floor is visible; birds stand apart with clear gaps between them.",
    2: "Moderate: birds cover about half of the floor, in loose groups with open floor between them.",
    3: "Dense: birds cover most of the floor, but open patches of floor remain between the groups.",
    4: (
        "Full: the floor is covered almost everywhere; only small gaps show between the birds, "
        "and they still have room to turn and walk."
    ),
    5: "Crammed: no floor shows anywhere; bodies are pressed tight against each other with no room to move.",
}

SYSTEM = (
    "You rate how densely chickens fill the floor in pictures from poultry shed cameras.\n"
    "Every picture shows chickens on a shed floor, even when the camera's colours are off (bluish or "
    "purple light) or the birds look small; never answer that there are no chickens.\n"
    "The shaded part of a picture is not monitored: judge the bright, unshaded area as a whole.\n"
    "Rate density only: how much of the floor the birds cover and how tightly they are pressed, "
    "not whether birds climb on each other.\n"
    "Use exactly one of these levels:\n"
    + "\n".join(f"{n}. {text}" for n, text in LEVELS.items())
    + '\nAnswer with JSON only: {"level": <1-5>, "reason": "<at most 12 words>"}'
)
PROMPT = "Rate this picture."

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_reply(text: str) -> tuple[int, str]:
    """(level, reason) from a model's reply; raises ValueError when it is not one."""
    match = _JSON_RE.search(text or "")
    if not match:
        raise ValueError(f"no JSON in reply {text!r}")
    data = json.loads(match.group(0))
    level = int(data["level"])
    if level not in LEVELS:
        raise ValueError(f"level {level} is not 1-5")
    return level, str(data.get("reason", "")).strip()[:200]
