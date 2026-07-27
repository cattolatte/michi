"""Tests for the bootstrap that produces michi's confidence intervals.

An interval is the one number in michi that users take on trust, so the ways
it can quietly go wrong are worth pinning: sharing resamples across metrics
must not change results, and the large-sample shortcut must not distort the
width in either direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from michi.evaluation.metrics import (
    BOOTSTRAP_MAX_ROWS,
    _rescale_interval,
    classification_metrics,
)


def _labels(rows: int, accuracy: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, rows)
    predictions = np.where(rng.random(rows) < accuracy, truth, 1 - truth)
    return truth, predictions


# --- the interval means something -----------------------------------------


def test_the_interval_brackets_the_estimate() -> None:
    """An interval that excludes its own point estimate is nonsense."""
    truth, predictions = _labels(2_000, 0.8)
    for metric in classification_metrics(truth, predictions, bootstrap=200):
        if metric.has_interval:
            assert metric.ci_low <= metric.value <= metric.ci_high


def test_a_larger_sample_gives_a_narrower_interval() -> None:
    """Uncertainty must fall as evidence accumulates."""
    small = classification_metrics(*_labels(1_000, 0.8), bootstrap=300)[1]
    large = classification_metrics(*_labels(16_000, 0.8), bootstrap=300)[1]
    assert small.has_interval and large.has_interval
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_the_interval_shrinks_at_roughly_the_square_root_rate() -> None:
    """Sixteen times the data should roughly quarter the interval.

    This is the property the large-sample rescaling relies on; if it did not
    hold here, the correction would be unfounded.
    """
    small = classification_metrics(*_labels(1_000, 0.8), bootstrap=400)[1]
    large = classification_metrics(*_labels(16_000, 0.8), bootstrap=400)[1]
    ratio = (small.ci_high - small.ci_low) / (large.ci_high - large.ci_low)
    assert 2.5 < ratio < 6.0


def test_disabling_the_bootstrap_leaves_a_bare_estimate() -> None:
    """`--bootstrap 0` reports the value without inventing an interval."""
    metrics = classification_metrics(*_labels(2_000, 0.8), bootstrap=0)
    assert all(not metric.has_interval for metric in metrics)


def test_too_few_rows_produces_no_interval() -> None:
    """Below a handful of rows, a percentile interval would be theatre."""
    metrics = classification_metrics(*_labels(8, 0.8), bootstrap=200)
    assert all(not metric.has_interval for metric in metrics)


# --- shared resamples ------------------------------------------------------


def test_every_metric_is_scored_on_the_same_resamples() -> None:
    """Sharing draws across metrics is an optimisation, not a change.

    Each metric used to draw its own resamples from an identically seeded
    generator, so the results were the same and the work was repeated. The
    shared path must reproduce that.
    """
    truth, predictions = _labels(3_000, 0.75)
    first = classification_metrics(truth, predictions, bootstrap=150, seed=11)
    second = classification_metrics(truth, predictions, bootstrap=150, seed=11)
    assert [m.ci_low for m in first] == [m.ci_low for m in second]


def test_the_seed_controls_the_interval() -> None:
    """Intervals are reproducible, which is what makes them citable."""
    truth, predictions = _labels(3_000, 0.75)
    a = classification_metrics(truth, predictions, bootstrap=150, seed=1)[1]
    b = classification_metrics(truth, predictions, bootstrap=150, seed=2)[1]
    assert a.value == b.value
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


# --- the large-sample correction -------------------------------------------


def test_rescaling_narrows_an_interval_from_a_partial_draw() -> None:
    """An interval from fewer rows is too wide and must be scaled in."""
    low, high = _rescale_interval(
        np.array([0.40, 0.60]), 0.50, drawn=25_000, total=400_000
    )
    assert 0.50 - low == pytest.approx(0.10 * (25_000 / 400_000) ** 0.5)
    assert high - 0.50 == pytest.approx(0.10 * (25_000 / 400_000) ** 0.5)


def test_rescaling_is_a_no_op_on_a_full_draw() -> None:
    """When every row was available, the interval stands as measured."""
    assert _rescale_interval(np.array([0.4, 0.6]), 0.5, drawn=100, total=100) == (
        0.4,
        0.6,
    )


def test_rescaling_preserves_asymmetry() -> None:
    """A lopsided interval stays lopsided; each side scales about the estimate."""
    low, high = _rescale_interval(
        np.array([0.30, 0.55]), 0.50, drawn=2_500, total=10_000
    )
    assert (0.50 - low) > (high - 0.50)


def test_a_large_sample_interval_is_not_inflated_by_the_shortcut() -> None:
    """The shortcut must not overstate uncertainty.

    Without the square-root correction, capping the draw reports an interval
    nearly three times too wide — which for a tool that sells honest
    uncertainty is as wrong as understating it. The corrected interval is
    compared against the analytic width for a proportion.
    """
    rows = BOOTSTRAP_MAX_ROWS * 8
    truth, predictions = _labels(rows, 0.75)

    accuracy = classification_metrics(truth, predictions, bootstrap=300)[1]
    assert accuracy.has_interval

    observed = accuracy.ci_high - accuracy.ci_low
    analytic = 2 * 1.96 * (accuracy.value * (1 - accuracy.value) / rows) ** 0.5
    assert observed == pytest.approx(analytic, rel=0.35)
