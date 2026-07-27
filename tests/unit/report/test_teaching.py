"""Tests for the teaching notes attached to a benchmark's own numbers.

The claim these notes make is narrow and worth defending: they describe *this*
result rather than statistics in general, and they never cross from describing
into advising. Both are testable.
"""

from __future__ import annotations

import pytest

from michi.bench.runner import BenchResult, ModelResult
from michi.bench.significance import Comparison
from michi.core.manifest import Metric
from michi.report.teaching import teaching_notes


def _model(
    name: str, scores: tuple[float, ...], *, metric: str = "balanced_accuracy"
) -> ModelResult:
    mean = sum(scores) / len(scores)
    return ModelResult(
        name=name,
        metrics=(Metric(name=metric, value=mean, greater_is_better=metric != "rmse"),),
        fold_scores=scores,
        fit_seconds=0.1,
    )


def _result(
    models: tuple[ModelResult, ...],
    comparisons: tuple[Comparison, ...] = (),
    *,
    n_rows: int = 600,
    metric: str = "balanced_accuracy",
) -> BenchResult:
    from michi.bench.preprocess import PreparationPolicy

    return BenchResult(
        task="classification",
        target="purchased",
        folds=5,
        primary_metric=metric,
        results=models,
        comparisons=comparisons,
        policy=PreparationPolicy(),
        n_rows=n_rows,
    )


# --- the baseline note -----------------------------------------------------


def test_the_gap_over_the_baseline_is_stated_not_the_headline_score() -> None:
    """What the modelling bought is the gap, and that is the number given."""
    result = _result(
        (
            _model("linear", (0.9, 0.88, 0.9, 0.89, 0.9)),
            _model("dummy", (0.5, 0.5, 0.5, 0.5, 0.5)),
        )
    )
    joined = " ".join(teaching_notes(result))
    assert "0.5" in joined
    assert "what the modelling bought" in joined


def test_a_model_that_loses_to_the_baseline_is_said_plainly() -> None:
    """No hedging when the features turned out to carry nothing."""
    result = _result(
        (
            _model("linear", (0.48, 0.5, 0.49, 0.51, 0.5)),
            _model("dummy", (0.5, 0.5, 0.5, 0.5, 0.5)),
        )
    )
    assert "No model beat the dummy baseline" in " ".join(teaching_notes(result))


# --- the spread note -------------------------------------------------------


def test_the_interval_is_explained_by_the_folds_that_produced_it() -> None:
    """The point is why *this* interval is *this* wide."""
    result = _result((_model("linear", (0.80, 0.95, 0.86, 0.91, 0.88)),))
    joined = " ".join(teaching_notes(result))
    assert "0.8" in joined and "0.95" in joined
    assert "120 rows" in joined  # 600 / 5 folds


def test_identical_fold_scores_produce_no_spread_note() -> None:
    """Silence beats a paragraph explaining a spread of zero."""
    result = _result((_model("linear", (0.9, 0.9, 0.9, 0.9, 0.9)),))
    assert not any("spread" in note for note in teaching_notes(result))


# --- the tie note ----------------------------------------------------------


def _tied_result(n_rows: int = 600) -> BenchResult:
    return _result(
        (
            _model("linear", (0.90, 0.88, 0.91, 0.89, 0.90)),
            _model("rf", (0.88, 0.89, 0.87, 0.90, 0.88)),
            _model("dummy", (0.5, 0.5, 0.5, 0.5, 0.5)),
        ),
        (
            Comparison("linear", "linear", 0.0, 1.0, 1.0, False),
            Comparison("rf", "linear", 0.014, 0.589, 0.589, False),
            Comparison("dummy", "linear", 0.39, 0.0, 0.0, True),
        ),
        n_rows=n_rows,
    )


def test_a_tie_is_explained_as_noise_exceeding_the_gap() -> None:
    """The reader should learn why the p-value says what it says."""
    joined = " ".join(teaching_notes(_tied_result()))
    assert "smaller than the disagreement" in joined
    assert "p=0.589" in joined


def test_the_rows_needed_to_separate_a_tie_are_computed() -> None:
    """A concrete number is the thing a textbook cannot give you."""
    joined = " ".join(teaching_notes(_tied_result()))
    assert "Back-of-envelope" in joined
    assert "rows, against the 600 here" in joined


def test_the_sample_size_estimate_scales_with_the_data_it_is_given() -> None:
    """It is arithmetic on this run, not a constant dressed up as one."""
    small = " ".join(teaching_notes(_tied_result(n_rows=600)))
    large = " ".join(teaching_notes(_tied_result(n_rows=6000)))
    assert small != large


def test_only_one_tie_is_worked_through() -> None:
    """One example teaches; five is a wall of text nobody reads."""
    result = _result(
        (
            _model("linear", (0.90, 0.88, 0.91, 0.89, 0.90)),
            _model("rf", (0.88, 0.89, 0.87, 0.90, 0.88)),
            _model("hist-gbm", (0.87, 0.88, 0.89, 0.88, 0.87)),
            _model("dummy", (0.5,) * 5),
        ),
        (
            Comparison("rf", "linear", 0.014, 0.589, 0.589, False),
            Comparison("hist-gbm", "linear", 0.019, 0.61, 0.61, False),
        ),
    )
    notes = teaching_notes(result)
    assert sum("swapped places" in note for note in notes) == 1


# --- the line it must not cross --------------------------------------------


def test_the_notes_describe_and_never_advise() -> None:
    """michi automates implementation, never judgement.

    A note that says "you should collect more data" has made the user's
    decision for them, which is the one thing this project does not do. The
    same sentence phrased as "separating these would take roughly N rows" is
    arithmetic the user can act on however they like.
    """
    advice = (
        "you should",
        "we recommend",
        "recommended",
        "the best choice",
        "you need to",
        "is better",
        "prefer ",
    )
    for note in teaching_notes(_tied_result()):
        lowered = note.lower()
        assert not any(phrase in lowered for phrase in advice), note


def test_an_uncertain_estimate_carries_its_caveat() -> None:
    """An unqualified number gets quoted; this one must not be unqualified."""
    estimate = next(
        note for note in teaching_notes(_tied_result()) if "Back-of-envelope" in note
    )
    assert "optimistic" in estimate
    assert "not a target" in estimate


# --- degenerate results ----------------------------------------------------


def test_a_result_where_every_model_failed_says_nothing() -> None:
    """No leader, no numbers to read."""
    failed = ModelResult(
        name="linear", metrics=(), fold_scores=(), fit_seconds=0.0, failed="boom"
    )
    assert teaching_notes(_result((failed,))) == ()


def test_a_single_model_still_explains_its_own_spread() -> None:
    """There is no comparison, but there is still an interval to account for."""
    result = _result((_model("linear", (0.80, 0.95, 0.86, 0.91, 0.88)),))
    assert teaching_notes(result)


@pytest.mark.parametrize("rows", [0, 1])
def test_a_degenerate_row_count_does_not_crash(rows: int) -> None:
    """Arithmetic on an empty dataset must not raise on the way out."""
    result = _result((_model("linear", (0.9, 0.88, 0.9, 0.89, 0.9)),), n_rows=rows)
    assert isinstance(teaching_notes(result), tuple)


# --- metrics that improve downward ----------------------------------------


def test_a_lower_is_better_metric_is_not_reported_backwards() -> None:
    """RMSE improves downward; subtracting one fixed way inverts the verdict.

    Reporting "the features bought 0.4 of rmse" when the model is *worse*
    than the baseline is the most misleading sentence this module could
    produce, so the direction comes from the metric rather than a guess.
    """
    result = _result(
        (
            _model("linear", (2.0, 2.1, 1.9, 2.0, 2.0), metric="rmse"),
            _model("dummy", (3.0, 3.1, 2.9, 3.0, 3.0), metric="rmse"),
        ),
        metric="rmse",
    )
    joined = " ".join(teaching_notes(result))
    assert "No model beat" not in joined
    assert "what the modelling bought" in joined


def test_a_baseline_that_wins_is_not_compared_against_itself() -> None:
    """When the dummy leads, "the best of the rest" would name the dummy."""
    result = _result(
        (
            _model("dummy", (2.0, 2.1, 1.9, 2.0, 2.0), metric="rmse"),
            _model("linear", (3.0, 3.1, 2.9, 3.0, 3.0), metric="rmse"),
        ),
        metric="rmse",
    )
    note = teaching_notes(result)[0]
    assert "The dummy baseline won" in note
    assert "dummy, the best of the rest" not in note


def test_a_tie_says_worse_or_better_not_above_or_below() -> None:
    """ "Above" reads as praise, and on RMSE it means the opposite."""
    result = _result(
        (
            _model("linear", (2.00, 2.10, 1.90, 2.00, 2.00), metric="rmse"),
            _model("rf", (2.05, 2.02, 2.11, 1.98, 2.06), metric="rmse"),
            _model("dummy", (3.0,) * 5, metric="rmse"),
        ),
        (
            Comparison("rf", "linear", 0.03, 0.6, 0.6, False),
            Comparison("dummy", "linear", 1.0, 0.001, 0.001, True),
        ),
        metric="rmse",
    )
    tie = next(note for note in teaching_notes(result) if "swapped places" in note)
    assert "above" not in tie and "below" not in tie
    assert "worse than" in tie or "better than" in tie
