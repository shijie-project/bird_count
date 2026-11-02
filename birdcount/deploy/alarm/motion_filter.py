from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass

from birdcount.deploy.alarm.models import CountSample, MotionFilterConfig


@dataclass(frozen=True)
class MotionDecision:
    excluded: bool
    reason: str = ""
    velocity_x_norm_per_sec: float | None = None
    velocity_y_norm_per_sec: float | None = None
    speed_norm_per_sec: float | None = None
    net_displacement_norm: float | None = None
    direction_degrees: float | None = None
    direction_consistency: float | None = None
    sample_count: int = 0
    window_duration_sec: float | None = None


# A drift needs two steps to have a direction: over a single step the path is
# the displacement, so any move would read as perfectly consistent. Structural,
# not a rate: at 1 fps the window has 3-4 points, at 5 fps 15.
MIN_POINTS = 3
# Frame timestamps jitter by milliseconds; a window boundary should not hinge on it.
_CLOCK_SLACK = 1e-3


class DirectionalMotionFilter:
    """Exclude dense but directional movement using density centroid drift."""

    def __init__(self, config: MotionFilterConfig):
        self.config = config
        self._history: dict[str, deque[CountSample]] = defaultdict(deque)

    def evaluate(self, sample: CountSample, threshold: float) -> MotionDecision:
        history = self._history[sample.camera_id]
        history.append(sample)
        min_ts = sample.timestamp - self.config.window_seconds
        while history and history[0].timestamp < min_ts:
            history.popleft()

        if not self.config.enabled:
            return MotionDecision(False, reason="disabled")
        if sample.count < threshold:
            return MotionDecision(False, reason="below_threshold")
        if sample.centroid_x is None or sample.centroid_y is None:
            return MotionDecision(False, reason="missing_centroid")

        points = [item for item in history if item.centroid_x is not None and item.centroid_y is not None]
        span = points[-1].timestamp - points[0].timestamp
        if len(points) < MIN_POINTS or span < self.config.min_span_seconds - _CLOCK_SLACK:
            return MotionDecision(False, reason="not_enough_history")

        smoothed = smooth_points(points, self.config.smoothing_seconds)
        first = smoothed[0]
        last = smoothed[-1]
        dt = last[0] - first[0]
        if dt <= 0:
            return MotionDecision(False, reason="zero_time_window")

        net_dx = last[1] - first[1]
        net_dy = last[2] - first[2]
        net = math.hypot(net_dx, net_dy)
        path = 0.0
        for prev, cur in itertools.pairwise(smoothed):
            path += math.hypot(
                cur[1] - prev[1],
                cur[2] - prev[2],
            )
        velocity_x = net_dx / dt
        velocity_y = net_dy / dt
        speed = net / dt
        consistency = net / path if path > 1e-9 else 0.0
        direction_degrees = (math.degrees(math.atan2(net_dy, net_dx)) + 360.0) % 360.0

        excluded = (
            net >= self.config.min_net_displacement_norm
            and speed >= self.config.min_speed_norm_per_sec
            and consistency >= self.config.min_direction_consistency
        )
        return MotionDecision(
            excluded=excluded,
            reason="directional_motion" if excluded else "not_directional_motion",
            velocity_x_norm_per_sec=velocity_x,
            velocity_y_norm_per_sec=velocity_y,
            speed_norm_per_sec=speed,
            net_displacement_norm=net,
            direction_degrees=direction_degrees,
            direction_consistency=consistency,
            sample_count=len(points),
            window_duration_sec=dt,
        )


def smooth_points(points: list[CountSample], smoothing_seconds: float) -> list[tuple[float, float, float]]:
    """Each centroid averaged with the others within `smoothing_seconds` / 2 of it (centred, in time)."""
    half = max(0.0, smoothing_seconds) / 2 + _CLOCK_SLACK
    smoothed: list[tuple[float, float, float]] = []
    lo = hi = 0
    for point in points:
        while points[lo].timestamp < point.timestamp - half:
            lo += 1
        while hi < len(points) and points[hi].timestamp <= point.timestamp + half:
            hi += 1
        chunk = points[lo:hi]
        cx = sum(float(item.centroid_x) for item in chunk) / len(chunk)
        cy = sum(float(item.centroid_y) for item in chunk) / len(chunk)
        smoothed.append((point.timestamp, cx, cy))
    return smoothed
