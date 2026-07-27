"""Tests for the text and time-series recipe operations, and the Tier 2 flags.

Text and time are the two shapes michi could profile but not use: a free-text
column could only be dropped, and a series had no way to look backwards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from michi.cli.app import app
from michi.core.errors import RecipeError
from michi.recipes import Recipe, RecipeStep, apply_recipe

runner = CliRunner()


def _apply(step: RecipeStep, frame: pd.DataFrame) -> pd.DataFrame:
    return apply_recipe(Recipe(steps=(step,)), frame, strict=False).frame


@pytest.fixture
def series() -> pd.DataFrame:
    """A small time series with a text column."""
    return pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=6, freq="D").astype(str),
            "store": ["a", "a", "a", "b", "b", "b"],
            "note": ["fast good", "slow", "great value here", "poor", "", "ok fine"],
            "sales": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        }
    )


# --- text -------------------------------------------------------------------


def test_text_length_counts_characters_and_words(series: pd.DataFrame) -> None:
    """Often most of the signal: how much someone wrote."""
    result = _apply(RecipeStep("text-length", {"columns": ["note"]}), series)
    assert result["note_words"].tolist()[:3] == [2.0, 1.0, 3.0]
    assert result["note_chars"].iloc[0] == 9


def test_an_empty_string_measures_as_zero_words(series: pd.DataFrame) -> None:
    """Blank is a length, not a missing value."""
    result = _apply(RecipeStep("text-length", {"columns": ["note"]}), series)
    assert result["note_words"].iloc[4] == 0


def test_tfidf_replaces_the_column_with_terms(series: pd.DataFrame) -> None:
    """A text column a model cannot read becomes columns it can."""
    result = _apply(
        RecipeStep("tfidf", {"columns": ["note"], "max_features": 5}), series
    )
    assert "note" not in result.columns
    assert any(name.startswith("note_tf_") for name in result.columns)


def test_tfidf_is_fitted_because_it_learns_a_vocabulary() -> None:
    """Fitting on a whole file lets the test fold vote on which words exist."""
    assert RecipeStep("tfidf", {"columns": ["note"]}).is_fitted
    assert not RecipeStep("text-length", {"columns": ["note"]}).is_fitted


def test_a_column_with_no_usable_terms_is_left_alone() -> None:
    """An empty vocabulary is not an error; the column carries no terms."""
    frame = pd.DataFrame({"note": ["", "", ""]})
    result = _apply(RecipeStep("tfidf", {"columns": ["note"]}), frame)
    assert len(result) == 3


# --- time series ------------------------------------------------------------


def test_a_lag_reads_only_earlier_rows(series: pd.DataFrame) -> None:
    """That is what makes it safe outside the fold: no future reaches the past."""
    result = _apply(
        RecipeStep("lag", {"columns": ["sales"], "by": "when", "periods": 1}), series
    )
    assert pd.isna(result["sales_lag1"].iloc[0])
    assert result["sales_lag1"].iloc[1] == 10.0


def test_a_lag_can_stay_inside_a_group(series: pd.DataFrame) -> None:
    """Store b's first row must not inherit store a's last."""
    result = _apply(
        RecipeStep(
            "lag",
            {"columns": ["sales"], "by": "when", "periods": 1, "group": "store"},
        ),
        series,
    )
    assert pd.isna(result["sales_lag1"].iloc[3])


def test_a_rolling_window_covers_the_rows_up_to_this_one(
    series: pd.DataFrame,
) -> None:
    """Including the current row, excluding every later one."""
    result = _apply(
        RecipeStep(
            "rolling",
            {"columns": ["sales"], "by": "when", "window": 2, "stat": "mean"},
        ),
        series,
    )
    assert result["sales_roll2_mean"].iloc[1] == pytest.approx(15.0)


def test_lag_and_rolling_refuse_to_guess_an_order(series: pd.DataFrame) -> None:
    """ "Earlier" needs a definition, and michi will not invent one."""
    with pytest.raises(RecipeError, match="by"):
        _apply(RecipeStep("lag", {"columns": ["sales"]}), series)


def test_a_rolling_window_of_one_is_refused(series: pd.DataFrame) -> None:
    """A window of one is the column itself."""
    with pytest.raises(RecipeError, match="at least 2"):
        _apply(
            RecipeStep("rolling", {"columns": ["sales"], "by": "when", "window": 1}),
            series,
        )


def test_time_ops_round_trip_through_their_command(tmp_path: Path) -> None:
    """Flag parity holds for the newest operations too."""
    from michi.recipes.author import command_for, recipe_from_flags

    recipe = recipe_from_flags(
        None,
        lag=[("sales", "1")],
        rolling=[("sales", "7:mean")],
        text_length=["note"],
        order_by="when",
    )
    command = command_for(recipe, "d.csv")
    for fragment in (
        "--lag sales=1",
        "--rolling sales=7:mean",
        "--text-length note",
        "--order-by when",
    ):
        assert fragment in command


def test_lag_without_an_order_column_is_refused_at_authoring() -> None:
    """Better to fail writing the recipe than when applying it."""
    from michi.recipes.author import recipe_from_flags

    with pytest.raises(RecipeError, match="--order-by"):
        recipe_from_flags(None, lag=[("sales", "1")])


# --- out-of-fold, balance, calibration --------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = 200
    signal = rng.normal(size=rows)
    pd.DataFrame(
        {
            "signal": signal,
            "label": (signal + rng.normal(scale=0.4, size=rows) > 0).astype(int),
        }
    ).to_csv(tmp_path / "d.csv", index=False)
    return tmp_path / "d.csv"


def test_out_of_fold_predictions_cover_every_row(dataset: Path, tmp_path: Path) -> None:
    """Each row predicted by a fold that did not train on it — the whole point.

    In-sample predictions stacked on produce a meta-model trained on the base
    models' memory rather than their generalisation.
    """
    out = tmp_path / "oof.csv"
    result = runner.invoke(
        app,
        [
            "bench",
            str(dataset),
            "--target",
            "label",
            "--models",
            "linear",
            "--cv",
            "3",
            "--no-save",
            "--oof",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    frame = pd.read_csv(out)
    assert len(frame) == 200
    assert frame["linear"].notna().all()
    assert "label" in frame.columns


def test_balancing_is_accepted_and_ignored_where_it_does_not_apply(
    dataset: Path,
) -> None:
    """A model without class_weight is not an error — the request just misses."""
    result = runner.invoke(
        app,
        [
            "bench",
            str(dataset),
            "--target",
            "label",
            "--models",
            "linear,knn",
            "--cv",
            "3",
            "--no-save",
            "--balance",
        ],
    )
    assert result.exit_code == 0, result.output


def test_calibration_wraps_the_model(dataset: Path, tmp_path: Path) -> None:
    """`eval` could report overconfidence and offered no way to fix it."""
    import joblib

    out = tmp_path / "cal.joblib"
    result = runner.invoke(
        app,
        [
            "fit",
            str(dataset),
            "--target",
            "label",
            "--model",
            "tree",
            "--calibrate",
            "isotonic",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Calibrated" in type(joblib.load(out)).__name__


def test_an_unknown_calibration_method_names_the_two(
    dataset: Path, tmp_path: Path
) -> None:
    """And says how they differ, since that is the whole choice."""
    result = runner.invoke(
        app,
        [
            "fit",
            str(dataset),
            "--target",
            "label",
            "--calibrate",
            "magic",
            "-o",
            str(tmp_path / "x.joblib"),
        ],
    )
    assert result.exit_code == 2
    combined = " ".join(result.output.split())
    assert "isotonic" in combined and "sigmoid" in combined
