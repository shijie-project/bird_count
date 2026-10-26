from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from .models import CountSample, MotionFilterConfig


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
        if len(points) < self.config.min_samples:
            return MotionDecision(False, reason="not_enough_samples")

        smoothed = smooth_points(points, self.config.smoothing_samples)
        first = smoothed[0]
        last = smoothed[-1]
        dt = last[0] - first[0]
        if dt <= 0:
            return MotionDecision(False, reason="zero_time_window")

        net_dx = last[1] - first[1]
        net_dy = last[2] - first[2]
        net = math.hypot(net_dx, net_dy)
        path = 0.0
        for prev, cur in zip(smoothed, smoothed[1:]):
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


def smooth_points(points: list[CountSample], smoothing_samples: int) -> list[tuple[float, float, float]]:
    window = max(1, int(smoothing_samples))
    half = window // 2
    smoothed: list[tuple[float, float, float]] = []
    for idx, point in enumerate(points):
        lo = max(0, idx - half)
        hi = min(len(points), idx + half + 1)
        chunk = points[lo:hi]
        cx = sum(float(item.centroid_x) for item in chunk if item.centroid_x is not None) / len(chunk)
        cy = sum(float(item.centroid_y) for item in chunk if item.centroid_y is not None) / len(chunk)
        smoothed.append((point.timestamp, cx, cy))
    return smoothed
