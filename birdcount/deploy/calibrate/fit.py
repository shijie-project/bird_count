"""From ratings to a threshold: the count at which a camera's pictures turn dense.

Per camera, the ratings fall into 10-bird bins; each bin has a share of
pictures rated dense (DENSE_LEVEL or more). That share should only rise with
the count, so it is made monotone (isotonic regression, pool-adjacent-
violators, weighted by the bin's ratings). The suggested threshold is where
the monotone share crosses TARGET, interpolated between bin centres and
rounded to 5 birds.

A suggestion needs both sides seen: at least MIN_SIDE dense and MIN_SIDE
not-dense ratings. Its error rate (dense pictures below it plus not-dense
pictures above it) says how well the count alone separates the two on this
camera; a high one means a threshold on the count cannot do the job there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from birdcount.deploy.calibrate.rubric import DENSE_LEVEL
from birdcount.deploy.countlog import SAMPLE_STEP


TARGET = 0.5  # share of dense pictures at which a camera should alarm
MIN_SIDE = 5  # ratings needed on each side (dense / not dense) for a suggestion
POOR_SEPARATION = 0.25  # an error rate above this: the count does not separate dense from not
MAX_STEP = 0.10  # an applied change moves the threshold at most this fraction at a time
ROUND_TO = 5


@dataclass
class Proposal:
    mac: str
    threshold: float  # in force now
    ratings: int = 0
    dense: int = 0
    bins: list[dict] = field(default_factory=list)  # {lo, n, dense, share, fitted}
    suggested: float | None = None
    step: float | None = None  # where the next applied change would put it
    error_now: float | None = None
    error_suggested: float | None = None
    alarm_pictures: int = 0
    alarm_dense: int = 0
    labels_compared: int = 0
    labels_agree: int = 0
    note: str = ""


def isotonic(values: list[float], weights: list[float]) -> list[float]:
    """Pool-adjacent-violators: the closest non-decreasing sequence, by weighted least squares."""
    blocks = []  # [sum of w*v, sum of w, length]
    for v, w in zip(values, weights, strict=True):
        blocks.append([v * w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, w2, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += w2
            blocks[-1][2] += n
    out = []
    for s, w, n in blocks:
        out += [s / w] * n
    return out


def error_rate(points: list[tuple[float, int]], threshold: float) -> float:
    wrong = sum(1 for count, level in points if (level >= DENSE_LEVEL) != (count >= threshold))
    return wrong / len(points) if points else 0.0


def propose(
    mac: str,
    threshold: float,
    points: list[tuple[float, int]],
    alarm_levels: list[int] = (),
    label_pairs: list[tuple[str, int]] = (),
) -> Proposal:
    """`points`: (count, level) of rated count-log snapshots; `alarm_levels`: ratings of alarm
    pictures; `label_pairs`: (person's label, model's level) on the same pictures."""
    p = Proposal(mac=mac, threshold=threshold, ratings=len(points))
    p.dense = sum(1 for _, lv in points if lv >= DENSE_LEVEL)
    p.alarm_pictures = len(alarm_levels)
    p.alarm_dense = sum(1 for lv in alarm_levels if lv >= DENSE_LEVEL)
    judged = [(lab, lv) for lab, lv in label_pairs if lab in ("pileup", "normal")]
    p.labels_compared = len(judged)
    p.labels_agree = sum(1 for lab, lv in judged if (lab == "pileup") == (lv >= DENSE_LEVEL))
    if not points:
        p.note = "no ratings yet"
        return p
    by_bin: dict[int, list[int]] = {}
    for count, level in points:
        by_bin.setdefault(int(count // SAMPLE_STEP), []).append(level)
    lows = sorted(by_bin)
    shares = [sum(lv >= DENSE_LEVEL for lv in by_bin[b]) / len(by_bin[b]) for b in lows]
    fitted = isotonic(shares, [len(by_bin[b]) for b in lows])
    p.bins = [
        {
            "lo": b * SAMPLE_STEP,
            "n": len(by_bin[b]),
            "dense": sum(lv >= DENSE_LEVEL for lv in by_bin[b]),
            "share": round(s, 3),
            "fitted": round(f, 3),
        }
        for b, s, f in zip(lows, shares, fitted, strict=True)
    ]
    p.error_now = round(error_rate(points, threshold), 3)
    if p.dense < MIN_SIDE or p.ratings - p.dense < MIN_SIDE:
        side = "dense" if p.dense < MIN_SIDE else "not-dense"
        p.note = f"too few {side} ratings to suggest a threshold ({p.dense} dense of {p.ratings})"
        return p
    centres = [b * SAMPLE_STEP + SAMPLE_STEP / 2 for b in lows]
    first = next((i for i, f in enumerate(fitted) if f >= TARGET), None)
    if first is None:
        p.note = "never rated dense often enough at any count seen"
        return p
    if first == 0:
        crossing = lows[0] * SAMPLE_STEP  # dense from the lowest count seen: the threshold sits at or below it
    else:
        f0, f1 = fitted[first - 1], fitted[first]
        crossing = centres[first - 1] + (TARGET - f0) / (f1 - f0) * (centres[first] - centres[first - 1])
    p.suggested = float(max(ROUND_TO, round(crossing / ROUND_TO) * ROUND_TO))
    p.step = round(min(max(p.suggested, threshold * (1 - MAX_STEP)), threshold * (1 + MAX_STEP)), 1)
    p.error_suggested = round(error_rate(points, p.suggested), 3)
    if p.error_suggested > POOR_SEPARATION:
        p.note = "the count separates dense from not-dense poorly on this camera"
    return p
