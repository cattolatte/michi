"""Tests for `michi ensemble`.

The claim is narrow: combining models is only worth it if the combination
beats the best single member by more than the noise. These tests hold the
ensemble to the same standard as every other model, and hold michi to making
no choices on the user's behalf.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from michi.bench.ensemble import ENSEMBLE_NAME, run_ensemble
from michi.cli.app import app
from michi.core.errors import RunError
from michi.core.io import load_table

runner = CliRunner()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A learnable classification set, small enough to combine quickly."""
    rng = np.random.default_rng(0)
    rows = 180
    signal = rng.normal(size=rows)
    pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=rows),
            "label": (signal + rng.normal(scale=0.4, size=rows) > 0).astype(int),
        }
    ).to_csv(tmp_path / "data.csv", index=False)
    return tmp_path / "data.csv"


# --- the ensemble is judged like anything else -----------------------------


def test_the_ensemble_is_ranked_beside_its_own_members(dataset: Path) -> None:
    """The only useful question is whether combining beat the best single model.

    An ensemble reported alone is a number with nothing to be measured
    against, which is how a combination that gained nothing gets shipped.
    """
    result = run_ensemble(
        load_table(dataset),
        target="label",
        members=("linear", "tree"),
        folds=3,
    )
    names = {item.name for item in result.results}
    assert ENSEMBLE_NAME in names
    assert {"linear", "tree"} <= names
    assert "dummy" in names, "the floor is included for an ensemble too"


def test_the_ensemble_is_significance_tested(dataset: Path) -> None:
    """A tie must be reported as a tie, not as a win."""
    result = run_ensemble(
        load_table(dataset),
        target="label",
        members=("linear", "tree"),
        folds=3,
    )
    compared = {item.model for item in result.comparisons}
    assert ENSEMBLE_NAME in compared


def test_voting_and_stacking_both_produce_a_result(dataset: Path) -> None:
    """Two ways to combine, both scored the same way."""
    for method in ("stack", "vote"):
        result = run_ensemble(
            load_table(dataset),
            target="label",
            members=("linear", "tree"),
            method=method,
            folds=3,
        )
        ensemble = next(item for item in result.results if item.name == ENSEMBLE_NAME)
        assert ensemble.failed is None, f"{method}: {ensemble.failed}"


# --- michi chooses nothing --------------------------------------------------


def test_an_ensemble_of_one_is_refused(dataset: Path) -> None:
    """Combining one model is not combining."""
    with pytest.raises(RunError, match="at least 2"):
        run_ensemble(load_table(dataset), target="label", members=("linear",))


def test_a_repeated_member_is_not_counted_twice(dataset: Path) -> None:
    """`--models rf,rf` is a typo, not a two-member ensemble."""
    with pytest.raises(RunError, match="at least 2"):
        run_ensemble(load_table(dataset), target="label", members=("linear", "linear"))


def test_an_unknown_method_names_the_ones_that_exist(dataset: Path) -> None:
    """An error says what to do, not only what went wrong."""
    with pytest.raises(RunError, match="stack"):
        run_ensemble(
            load_table(dataset),
            target="label",
            members=("linear", "tree"),
            method="magic",
        )


# --- the transient registration must not leak ------------------------------


def test_the_ensemble_does_not_linger_in_the_catalogue(dataset: Path) -> None:
    """It is assembled from what the user named, so it cannot be permanent.

    A leftover entry would make `michi bench --models ensemble` mean whatever
    the last ensemble command happened to build.
    """
    from michi.bench import available_models

    run_ensemble(
        load_table(dataset), target="label", members=("linear", "tree"), folds=3
    )
    assert ENSEMBLE_NAME not in {entry.name for entry in available_models()}


def test_a_failed_run_still_cleans_up(dataset: Path) -> None:
    """The catalogue must be restored even when the benchmark raises."""
    from michi.bench import available_models

    with pytest.raises(Exception, match=r".*"):
        run_ensemble(
            load_table(dataset),
            target="not_a_column",
            members=("linear", "tree"),
            folds=3,
        )
    assert ENSEMBLE_NAME not in {entry.name for entry in available_models()}


# --- the command -----------------------------------------------------------


def test_the_command_renders_a_leaderboard(dataset: Path) -> None:
    """It reports as a benchmark because it is one."""
    result = runner.invoke(
        app,
        [
            "ensemble",
            str(dataset),
            "--target",
            "label",
            "--models",
            "linear,tree",
            "--cv",
            "3",
            "--no-save",
        ],
    )
    assert result.exit_code == 0, result.output
    combined = " ".join(result.output.split())
    assert "ensemble" in combined
    assert "dummy" in combined
