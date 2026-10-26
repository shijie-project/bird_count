from __future__ import annotations

import uuid
from datetime import UTC, datetime


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> float:
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def safe_name(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"


def new_event_id(camera_id: str, timestamp: float) -> str:
    """`<camera>_<UTC time>_<8 hex>`: unique, sortable, safe as a folder name."""
    stamp = utc_iso(timestamp).replace(":", "").replace("-", "")
    return f"{safe_name(camera_id)}_{stamp}_{uuid.uuid4().hex[:8]}"


def snapshot_name(event_id: str, action_type: str, timestamp: float) -> str:
    """File name of the frame saved for one alarm action."""
    return f"{safe_name(event_id)}_{safe_name(action_type)}_{int(timestamp)}.jpg"
