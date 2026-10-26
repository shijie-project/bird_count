"""The one fixed standard a picture is judged by: five crowding levels.

Every rating, by a model or compared against a person's label, uses this
text unchanged, so ratings from different days and runs mean the same thing.
Changing it starts a new standard: bump RUBRIC_VERSION, which is stored with
every rating, and old ratings stop counting for new proposals.
"""

from __future__ import annotations

import json
import re


RUBRIC_VERSION = 1
DENSE_LEVEL = 4  # a rating at or above this is "dense enough to alarm"

LEVELS = {
    1: "Sparse: most of the floor is visible; birds are spread out with clear gaps between them.",
    2: "Moderate: birds cover about half the floor, in loose groups with gaps.",
    3: "Dense: the floor is mostly covered, but birds stand separately and each body is distinct.",
    4: "Packed: the floor is fully covered and birds are pressed body to body with no gaps.",
    5: "Piled up: birds are on top of one another, in a heap or mound; single birds are hard to tell apart.",
}

SYSTEM = (
    "You rate how crowded the chickens are in pictures from a poultry shed camera, for a pile-up alarm.\n"
    "The shaded part of a picture is not monitored: judge only the bright, unshaded area.\n"
    "Use exactly one of these levels:\n"
    + "\n".join(f"{n}. {text}" for n, text in LEVELS.items())
    + "\nJudge the most crowded part of the monitored area, not the average. "
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
