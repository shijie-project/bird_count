"""The timeline figure: one panel per reported series, ready for a report.

matplotlib is imported here only, when a figure is drawn.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from birdcount.training.video.timeline import Sample, clock_of, rolling_mean


# (series, title, unit, colour); the order the panels stack in.
PANELS = (
    ("peak", "Max local flock count", "birds in the busiest {window_px}×{window_px} px patch", "#c2410c"),
    ("count", "Global flock count", "birds in frame", "#2563eb"),
)
THRESHOLD_COLOR = "#c93632"


def render_chart(
    path: Path,
    samples: list[Sample],
    *,
    title: str,
    report: str,
    start: datetime | None,
    thresholds: dict[str, float],
    smooth: int,
    window_px: int,
    interval: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    seconds = [s.t for s in samples]
    base = start or datetime(2000, 1, 1)  # elapsed time still goes on a date axis, sharing its formatter
    x = [base + timedelta(seconds=t) for t in seconds]
    panels = [p for p in PANELS if report in ("both", p[0])]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "figure.dpi": 200})
    height = 2.9 * len(panels) + 0.4
    figure, axes = plt.subplots(len(panels), 1, figsize=(9.2, height), sharex=True, squeeze=False)
    headline = []
    for axis, (key, name, unit, color) in zip(axes[:, 0], panels, strict=True):
        values = [getattr(s, key) for s in samples]
        axis.fill_between(x, values, color=color, alpha=0.13, linewidth=0)
        axis.plot(x, values, color=color, alpha=0.45, linewidth=0.9)
        if smooth > 1:
            axis.plot(x, rolling_mean(values, smooth), color=color, linewidth=1.9, solid_capstyle="round")
        threshold = thresholds.get(key, 0.0)
        top_value = max(max(values), threshold)
        if threshold > 0:
            axis.axhline(threshold, color=THRESHOLD_COLOR, linestyle=(0, (5, 4)), linewidth=1.0, alpha=0.8)
            axis.axhspan(threshold, top_value * 1.12, color=THRESHOLD_COLOR, alpha=0.05, linewidth=0)

        top = max(range(len(values)), key=values.__getitem__)
        axis.plot(x[top], values[top], "o", color=color, markersize=4.5, markeredgecolor="white", zorder=5)
        position = (seconds[top] - seconds[0]) / max(seconds[-1] - seconds[0], 1e-6)
        axis.annotate(
            f"{values[top]:.1f} @ {clock_of(start, seconds[top])}",
            xy=(x[top], values[top]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="left" if position < 0.15 else "right" if position > 0.85 else "center",  # never off the axis
            fontsize=8,
            color=color,
            fontweight="semibold",
        )
        headline.append(f"max {values[top]:.1f} ({'in one patch' if key == 'peak' else 'in frame'})")

        axis.set_title(name, loc="left", fontweight="semibold", color="#172033")
        axis.set_ylabel(unit.format(window_px=window_px), fontsize=8.5, color="#526176")
        axis.set_ylim(0, top_value * 1.18 or 1.0)
        axis.margins(x=0.01)
        axis.grid(axis="y", color="#e5eaf1", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#d7dee8")
        axis.tick_params(colors="#64748b", labelsize=8)

    axis = axes[-1, 0]
    span = seconds[-1] - seconds[0]
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M" if span > 900 else "%H:%M:%S"))
    axis.set_xlabel("clock time" if start is not None else "elapsed time", fontsize=8.5, color="#526176")
    figure.autofmt_xdate(rotation=0, ha="center")

    # Title and subtitle have a fixed point size, so their room is reserved in inches.
    figure.suptitle(
        f"Predicted density over time — {title}",
        x=0.012,
        y=0.985,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#172033",
    )
    figure.text(
        0.012,
        0.985 - 0.32 / height,
        f"{len(samples)} samples · every {interval:.2f}s · "
        f"{clock_of(start, seconds[0])}–{clock_of(start, seconds[-1])} · {', '.join(headline)}",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#748298",
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.62 / height))
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
