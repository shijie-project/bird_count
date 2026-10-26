"""The one fixed standard a picture is judged by: five crowding levels.

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
RUBRIC_VERSION = 2
DENSE_LEVEL = 5  # a rating at or above this is a pile-up, the thing the alarm is for

LEVELS = {
    1: "Sparse: most of the floor is visible; birds are spread out with clear gaps between them.",
    2: "Moderate: birds cover about half the floor, in loose groups with gaps.",
    3: "Dense: the floor is mostly covered; birds stand or sit side by side, each body distinct and on the floor.",
    4: (
        "Packed: the floor is fully covered and birds are pressed body to body, but every bird still "
        "stands on the floor. This is normal crowding in a full shed."
    ),
    5: (
        "Piled up: birds are on top of other birds: climbing onto backs, stacked in layers or heaped into "
        "a mound, often against a wall, a corner or an obstacle; you cannot tell which birds touch the floor."
    ),
}

SYSTEM = (
    "You rate how crowded the chickens are in pictures from a poultry shed camera, for a pile-up alarm.\n"
    "The shaded part of a picture is not monitored: judge only the bright, unshaded area.\n"
    "Use exactly one of these levels:\n"
    + "\n".join(f"{n}. {text}" for n, text in LEVELS.items())
    + "\nJudge the most crowded part of the monitored area, not the average. A full shed is normally "
    "packed (4): give 5 only when birds are on top of other birds. "
    'Answer with JSON only: {"level": <1-5>, "reason": "<at most 12 words>"}'
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
