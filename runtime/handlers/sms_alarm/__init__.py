"""Chicken pile-up SMS alarm handler.

Decision logic (motion filter + Level 1/2/3 + recovery state machine) lives in
the vendored `alarm/` package; this subpackage is the realtime wiring only.
"""

from ._handler import SmsAlarmHandler


__all__ = ["SmsAlarmHandler"]
