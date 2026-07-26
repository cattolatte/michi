"""Tests for significance testing.

The corrected resampled t-test is the piece that makes michi's comparisons
honest, so it is tested against known behaviour rather than fixed p-values.
"""

from __future__ import annotations

import numpy as np

from michi.bench.significance import (
    _holm,
    compare_to_leader,
    corrected_paired_t_test,
)

# --- the corrected test ----------------------------------------------------


def test_identical_scores_are_never_significant() -> None:
    """Two models with the same fold scores cannot be distinguished."""
    scores = np.array([0.8, 0.82, 0.79, 0.81, 0.80])
    assert corrected_paired_t_test(scores, scores, train_size=800, test_size=200) == 1.0


def test_a_large_consistent_gap_is_significant() -> None:
    """A model better on every fold by a wide margin is distinguishable."""
    strong = np.array([0.90, 0.91, 0.89, 0.92, 0.90])
    weak = np.array([0.50, 0.53, 0.47, 0.55, 0.49])
    assert corrected_paired_t_test(strong, weak, train_size=800, test_size=200) < 0.05


def test_a_perfectly_consistent_gap_is_significant() -> None:
    """An identical difference on every fold is the strongest evidence there is.

    A zero-variance difference sends the t-statistic to infinity, so the limit
    is p = 0. Treating it as "cannot tell" would invert the conclusion.
    """
    strong = np.array([0.90, 0.91, 0.89, 0.92, 0.90])
    weak = strong - 0.4
    assert corrected_paired_t_test(strong, weak, train_size=800, test_size=200) == 0.0


def test_a_tiny_noisy_gap_is_not_significant() -> None:
    """A small difference swamped by fold variance is reported as noise."""
    first = np.array([0.80, 0.70, 0.85, 0.65, 0.90])
    second = np.array([0.79, 0.72, 0.83, 0.68, 0.86])
    assert corrected_paired_t_test(first, second, train_size=800, test_size=200) > 0.05


def test_the_correction_is_more_conservative_than_a_naive_test() -> None:
    """Accounting for overlapping training sets must widen, never narrow.

    This is the property the whole module exists for: an uncorrected paired
    t-test over folds treats correlated evidence as independent and calls
    noise significant.
    """
    from scipy import stats

    first = np.array([0.82, 0.75, 0.88, 0.79, 0.84])
    second = np.array([0.78, 0.74, 0.80, 0.72, 0.81])

    corrected = corrected_paired_t_test(first, second, train_size=800, test_size=200)
    naive = float(stats.ttest_rel(first, second).pvalue)
    assert corrected > naive


def test_single_fold_cannot_support_a_test() -> None:
    """One fold gives no variance estimate, so nothing is claimed."""
    assert (
        corrected_paired_t_test(
            np.array([0.9]), np.array([0.5]), train_size=80, test_size=20
        )
        == 1.0
    )


# --- multiple comparisons --------------------------------------------------


def test_holm_adjustment_raises_p_values() -> None:
    """Adjusted p-values are never smaller than the raw ones."""
    raw = {"a": 0.01, "b": 0.02, "c": 0.04}
    adjusted = _holm(raw)
    assert all(adjusted[name] >= raw[name] for name in raw)


def test_holm_adjustment_is_monotone() -> None:
    """Ordering by significance survives the correction."""
    adjusted = _holm({"a": 0.001, "b": 0.01, "c": 0.5})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


# --- comparing a field of models -------------------------------------------


def test_leader_is_the_best_scoring_model() -> None:
    """The comparison is anchored on the highest score, for a greater-better metric."""
    scores = {
        "good": np.array([0.9, 0.91, 0.89]),
        "poor": np.array([0.5, 0.51, 0.49]),
    }
    comparisons = compare_to_leader(
        scores, greater_is_better=True, train_size=800, test_size=200
    )
    assert comparisons[0].model == "good"
    assert comparisons[0].leader == "good"


def test_leader_is_the_lowest_for_error_metrics() -> None:
    """For an error metric, lower wins — direction is never assumed."""
    scores = {
        "accurate": np.array([1.0, 1.1, 0.9]),
        "poor": np.array([5.0, 5.1, 4.9]),
    }
    comparisons = compare_to_leader(
        scores, greater_is_better=False, train_size=800, test_size=200
    )
    assert comparisons[0].model == "accurate"


def test_the_leader_is_never_marked_significant_against_itself() -> None:
    """A model cannot beat itself."""
    scores = {"a": np.array([0.9, 0.9, 0.9]), "b": np.array([0.5, 0.5, 0.5])}
    comparisons = compare_to_leader(
        scores, greater_is_better=True, train_size=800, test_size=200
    )
    leader = next(item for item in comparisons if item.model == item.leader)
    assert leader.significant is False


def test_verdict_reads_as_a_sentence() -> None:
    """The plain-language verdict is what users act on."""
    scores = {
        "strong": np.array([0.95, 0.94, 0.96]),
        "weak": np.array([0.40, 0.41, 0.39]),
    }
    comparisons = compare_to_leader(
        scores, greater_is_better=True, train_size=800, test_size=200
    )
    loser = next(item for item in comparisons if item.model == "weak")
    assert "worse than strong" in loser.verdict


def test_close_models_are_reported_as_indistinguishable() -> None:
    """Near-identical models are called tied, not ranked as if different."""
    scores = {
        "a": np.array([0.80, 0.72, 0.85, 0.66, 0.90]),
        "b": np.array([0.79, 0.73, 0.84, 0.67, 0.89]),
    }
    comparisons = compare_to_leader(
        scores, greater_is_better=True, train_size=800, test_size=200
    )
    challenger = next(item for item in comparisons if item.model != item.leader)
    assert challenger.significant is False
    assert "not distinguishable" in challenger.verdict


def test_empty_input_returns_no_comparisons() -> None:
    """Nothing to compare produces nothing, rather than failing."""
    assert (
        compare_to_leader({}, greater_is_better=True, train_size=1, test_size=1) == ()
    )
