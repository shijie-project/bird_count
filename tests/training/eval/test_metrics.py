import math

import numpy as np
import pytest

from birdcount.training.eval.metrics import (
    accuracy_grid,
    compute_metrics,
    compute_pileup_classification,
    compute_stratified,
    fraction_within,
    within_tolerance,
)


PREDS = np.array([10.0, 48.0, 95.0, 130.0])
GTS = np.array([12.0, 50.0, 100.0, 110.0])


def test_counting_metrics() -> None:
    m = compute_metrics(PREDS, GTS)
    assert m.mae == pytest.approx((2 + 2 + 5 + 20) / 4)
    assert m.bias == pytest.approx((-2 - 2 - 5 + 20) / 4)
    assert m.counting_accuracy == pytest.approx(1 - 29 / 272)
    assert m.rmse == pytest.approx(math.sqrt((4 + 4 + 25 + 400) / 4))
    assert m.worst_abs_error == 20 and m.best_abs_error == 2


def test_tolerance_rule_takes_the_looser_bound() -> None:
    # |err| <= max(5, 10% of gt): 2 ok, 2 ok, 5 ok (10 allowed), 20 > 11
    assert fraction_within(PREDS, GTS, abs_tol=5, rel_tol=0.10) == 0.75
    assert within_tolerance(-5, 100) and not within_tolerance(20, 110)
    grid = accuracy_grid(PREDS, GTS, (0.0, 5.0), (0.0, 0.10))
    assert grid[(5.0, 0.10)] == 0.75 and grid[(0.0, 0.0)] == 0.0


def test_pileup_confusion() -> None:
    c = compute_pileup_classification(PREDS, GTS, threshold=100)
    assert (c.tp, c.fp, c.tn, c.fn) == (1, 0, 3, 0)
    assert c.recall == 1.0


def test_stratified_bands() -> None:
    bands = {s.band: s for s in compute_stratified(PREDS, GTS)}
    assert bands["Medium (11-50)"].n_images == 2
    assert bands["Pile-up (>100)"].mae == 20
    assert bands["Empty"].n_images == 0 and bands["Empty"].mape is None
