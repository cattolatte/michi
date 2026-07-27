"""Tests for the inline SVG charts.

A chart is a rendering of an artifact, never a computation. These tests hold
that line, and hold the charts to refusing rather than misleading when the
data cannot be drawn honestly.
"""

from __future__ import annotations

from typing import Any

import pytest

from michi.report.charts import (
    calibration_chart,
    confusion_chart,
    interval_chart,
    slice_chart,
)

# --- confusion matrix ------------------------------------------------------


def test_a_confusion_matrix_draws_every_cell() -> None:
    """Two classes make four cells, and each carries its count."""
    svg = confusion_chart([0, 1], [[40, 5], [7, 48]])
    assert svg is not None
    for count in ("40", "5", "7", "48"):
        assert f">{count}</text>" in svg


def test_too_many_classes_draws_nothing() -> None:
    """A twenty-class grid is unreadable; the caller shows the table instead."""
    classes = list(range(20))
    assert confusion_chart(classes, [[1] * 20 for _ in range(20)]) is None


def test_a_ragged_matrix_is_refused() -> None:
    """Drawing a mismatched matrix would silently mislabel the axes."""
    assert confusion_chart([0, 1, 2], [[1, 2], [3, 4]]) is None


def test_shading_is_per_row_so_a_rare_class_stays_visible() -> None:
    """Whole-matrix shading makes the majority class the only visible thing.

    That is precisely the failure a confusion matrix is read to find, so a
    rare class that is always wrong must not fade to nothing.
    """
    svg = confusion_chart([0, 1], [[990, 0], [8, 2]])
    assert svg is not None
    opacities = [float(part.split('"')[0]) for part in svg.split('fill-opacity="')[1:]]
    # The rare class's dominant cell is shaded strongly despite being 8 of 1000.
    assert max(opacities) > 0.6
    assert sorted(opacities)[-2] > 0.5


# --- calibration -----------------------------------------------------------


def test_calibration_reads_the_triples_a_manifest_records() -> None:
    """Manifests store bins as [predicted, observed, count]."""
    svg = calibration_chart([[0.1, 0.08, 50], [0.5, 0.47, 80], [0.9, 0.93, 60]])
    assert svg is not None and "<path" in svg


def test_calibration_also_reads_a_mapping() -> None:
    """A hand-written or future artifact should still draw."""
    svg = calibration_chart(
        [
            {"predicted": 0.2, "observed": 0.18, "count": 10},
            {"predicted": 0.8, "observed": 0.75, "count": 12},
        ]
    )
    assert svg is not None


def test_one_bin_is_not_a_curve() -> None:
    """A single point drawn as a line implies a trend that was not measured."""
    assert calibration_chart([[0.5, 0.5, 100]]) is None
    assert calibration_chart(None) is None


# --- intervals -------------------------------------------------------------


def test_an_interval_chart_marks_the_leader() -> None:
    """The best score is coloured; the rest are not."""
    svg = interval_chart([("linear", 0.9, 0.85, 0.95), ("rf", 0.8, 0.74, 0.86)])
    assert svg is not None
    assert "linear" in svg and "rf" in svg


def test_an_interval_chart_survives_missing_bounds() -> None:
    """A metric without a bootstrap still has a point estimate worth showing."""
    svg = interval_chart([("accuracy", 0.9, None, None)])
    assert svg is not None and "0.9" in svg


def test_identical_scores_do_not_divide_by_zero() -> None:
    """Every model scoring the same is a real result, not a crash."""
    svg = interval_chart([("a", 0.5, 0.5, 0.5), ("b", 0.5, 0.5, 0.5)])
    assert svg is not None


# --- slices ----------------------------------------------------------------


def _slice(column: str, value: str, score: float, rows: int) -> dict[str, Any]:
    return {"column": column, "value": value, "score": score, "n_rows": rows}


def test_the_worst_subgroup_comes_first() -> None:
    """An average hides the group a model fails; it belongs at the top."""
    svg = slice_chart(
        [
            _slice("region", "north", 0.94, 200),
            _slice("region", "west", 0.61, 40),
            _slice("region", "east", 0.90, 150),
        ]
    )
    assert svg is not None
    assert svg.index("west") < svg.index("north")


def test_a_slice_score_is_read_from_score_not_value() -> None:
    """`value` is the subgroup's label; reading it as the score plots names."""
    svg = slice_chart([_slice("region", "north", 0.94, 200)])
    assert svg is not None
    assert "0.94" in svg


def test_no_scores_draws_nothing() -> None:
    """Slices without scores are not a chart."""
    assert slice_chart([{"column": "region", "value": "north"}]) is None
    assert slice_chart([]) is None


# --- the shared contract ---------------------------------------------------


@pytest.mark.parametrize(
    "svg",
    [
        confusion_chart([0, 1], [[4, 1], [2, 5]]),
        calibration_chart([[0.2, 0.2, 10], [0.7, 0.6, 12]]),
        interval_chart([("a", 0.5, 0.4, 0.6)]),
        slice_chart([_slice("g", "x", 0.5, 10)]),
    ],
)
def test_charts_inherit_the_page_colour(svg: str | None) -> None:
    """A hardcoded near-black is invisible on the dark viewer.

    These render into two documents with opposite backgrounds, so text and
    rules take the page's own colour rather than naming one — which is how
    they first shipped, and why nothing was legible in the viewer.
    """
    assert svg is not None
    assert "currentColor" in svg


@pytest.mark.parametrize(
    "svg",
    [
        confusion_chart([0, 1], [[4, 1], [2, 5]]),
        calibration_chart([[0.2, 0.2, 10], [0.7, 0.6, 12]]),
        interval_chart([("a", 0.5, 0.4, 0.6)]),
        slice_chart([_slice("g", "x", 0.5, 10)]),
    ],
)
def test_charts_fetch_nothing(svg: str | None) -> None:
    """No CDN, no script: the viewer must work air-gapped."""
    assert svg is not None
    assert "http" not in svg
    assert "<script" not in svg


def test_a_hostile_label_cannot_inject_markup() -> None:
    """Column names come from user data and are rendered into a document."""
    svg = slice_chart([_slice("<script>x</script>", "y", 0.5, 1)])
    assert svg is not None
    assert "<script>" not in svg
