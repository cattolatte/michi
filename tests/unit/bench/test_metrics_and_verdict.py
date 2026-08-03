"""Tests for user-chosen metrics, and for saying which way a metric improves.

A competition scores on its own metric; optimising anything else is climbing
the wrong hill. And a verdict that says "scores highest" about RMSE is the
same inversion the teaching notes carried until v1.2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from michi.bench import run_benchmark, scorers_for
from michi.core.errors import RunError
from michi.core.io import load_table


@pytest.fixture
def regression(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    x = rng.normal(size=240)
    pd.DataFrame({"x": x, "price": np.exp(2 + x * 0.5)}).to_csv(
        tmp_path / "reg.csv", index=False
    )
    return tmp_path / "reg.csv"


# --- choosing the metric ----------------------------------------------------


def test_a_named_metric_becomes_the_headline(regression: Path) -> None:
    """The head of the scorer tuple is what ranks, intervals, and tests."""
    result = run_benchmark(
        load_table(regression),
        target="price",
        models=("linear",),
        folds=3,
        metric="rmsle",
    )
    assert result.primary_metric == "rmsle"


def test_competition_metrics_are_built_in(regression: Path) -> None:
    """RMSLE and MAPE are common enough to ship rather than require a plugin."""
    names = [name for name, _, _ in scorers_for("regression")]
    assert "rmsle" in names
    assert "mape" in names


def test_rmsle_survives_a_negative_prediction() -> None:
    """A model undershooting to -0.001 must not fail the whole run."""
    scorers = {name: fn for name, fn, _ in scorers_for("regression")}
    value = scorers["rmsle"](np.array([1.0, 2.0]), np.array([-0.001, 2.0]))
    assert value == value  # not NaN


def test_mape_ignores_rows_whose_truth_is_zero() -> None:
    """Dividing by a true zero is undefined, not infinite."""
    scorers = {name: fn for name, fn, _ in scorers_for("regression")}
    value = scorers["mape"](np.array([0.0, 4.0]), np.array([1.0, 2.0]))
    assert value == pytest.approx(0.5)


def test_an_unknown_metric_names_what_exists_and_how_to_add_one() -> None:
    """An error says what to do, not only what went wrong."""
    with pytest.raises(RunError, match=r"michi\.metrics"):
        scorers_for("regression", "map_at_k")


def test_metric_direction_is_carried_not_guessed(regression: Path) -> None:
    """RMSLE improves downward, and every renderer needs to know that."""
    result = run_benchmark(
        load_table(regression),
        target="price",
        models=("linear",),
        folds=3,
        metric="rmsle",
    )
    leader = result.leader
    assert leader is not None
    assert leader.primary.greater_is_better is False


# --- the verdict says which way ---------------------------------------------


def test_a_lower_is_better_leader_does_not_score_highest(
    regression: Path,
) -> None:
    """ "Scores highest" about RMSE is simply false — the leader scores lowest."""
    from rich.console import Console

    from michi.report import render_benchmark

    result = run_benchmark(
        load_table(regression), target="price", models=("linear", "tree"), folds=3
    )
    console = Console(force_terminal=False, width=100)
    with console.capture() as captured:
        render_benchmark(result, console)
    output = " ".join(captured.get().split())
    assert "scores lowest" in output
    assert "scores highest" not in output


def test_a_greater_is_better_leader_still_scores_highest(tmp_path: Path) -> None:
    """The fix must not invert the case that was already correct."""
    from rich.console import Console

    from michi.report import render_benchmark

    rng = np.random.default_rng(0)
    signal = rng.normal(size=200)
    pd.DataFrame({"x": signal, "label": (signal > 0).astype(int)}).to_csv(
        tmp_path / "c.csv", index=False
    )

    result = run_benchmark(
        load_table(tmp_path / "c.csv"),
        target="label",
        models=("linear", "tree"),
        folds=3,
    )
    console = Console(force_terminal=False, width=100)
    with console.capture() as captured:
        render_benchmark(result, console)
    assert "scores highest" in " ".join(captured.get().split())
