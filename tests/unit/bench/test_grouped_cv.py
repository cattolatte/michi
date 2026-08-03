"""Tests for group-aware cross-validation.

`michi split --group` has prevented an entity spanning both sides of a split
since v1.9, while `bench` cross-validated without honouring groups at all —
michi contradicting its own advice, and overstating a score by 26 points on
the pattern the advice exists for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from michi.bench import run_benchmark
from michi.core.errors import DataError
from michi.core.io import load_table

runner = CliRunner()


@pytest.fixture
def leaky(tmp_path: Path) -> Path:
    """Rows whose label is a property of the entity, not of the features.

    Each customer appears five times with a customer-specific quirk. A model
    that memorises the quirk scores well under plain K-fold, because four of
    a customer's rows train it and the fifth is scored — and that score is
    memory rather than anything that will survive a new customer.
    """
    rng = np.random.default_rng(0)
    rows = []
    for customer in range(60):
        quirk = rng.normal() * 3
        label = int(rng.random() < 0.5)
        for _ in range(5):
            rows.append(
                {
                    "customer": f"c{customer}",
                    "quirk": quirk + rng.normal(scale=0.01),
                    "label": label,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "leaky.csv", index=False)
    return tmp_path / "leaky.csv"


def _score(path: Path, group: str | None) -> float:
    result = run_benchmark(
        load_table(path), target="label", models=("rf",), folds=5, group=group
    )
    leader = next(item for item in result.results if item.name == "rf")
    return float(leader.primary.value)


def test_grouping_removes_a_score_that_was_memory(leaky: Path) -> None:
    """The whole point: an entity must not span folds.

    Without grouping the model recognises the customer and reports far more
    skill than it has. The grouped score is the honest one, and the gap
    between them is what michi was previously hiding.
    """
    leaked = _score(leaky, group=None)
    honest = _score(leaky, group="customer")
    assert leaked > honest + 0.15, f"leaked={leaked}, honest={honest}"
    assert honest < 0.7


def test_the_grouping_column_is_not_used_as_a_feature(leaky: Path) -> None:
    """Otherwise the model reads the entity id directly and the fix is moot."""
    result = run_benchmark(
        load_table(leaky),
        target="label",
        models=("rf",),
        folds=5,
        group="customer",
    )
    manifest = result.manifests[0]
    assert "customer" not in manifest.details.get("features", [])


def test_a_missing_group_column_is_actionable(leaky: Path) -> None:
    """An error names what went wrong, not only that something did."""
    with pytest.raises(DataError, match="nonesuch"):
        run_benchmark(
            load_table(leaky),
            target="label",
            models=("rf",),
            folds=3,
            group="nonesuch",
        )


def test_folds_reduce_when_there_are_fewer_groups_than_folds(
    tmp_path: Path,
) -> None:
    """Splitting three groups five ways would have to split a group."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "team": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
            "x": rng.normal(size=60),
            "label": rng.integers(0, 2, 60),
        }
    )
    frame.to_csv(tmp_path / "few.csv", index=False)
    result = run_benchmark(
        load_table(tmp_path / "few.csv"),
        target="label",
        models=("tree",),
        folds=5,
        group="team",
    )
    assert result.folds <= 3


def test_grouping_reaches_bench_through_the_command(leaky: Path) -> None:
    """A capability the API has and the CLI does not is invisible to users."""
    from michi.cli.app import app

    result = runner.invoke(
        app,
        [
            "bench",
            str(leaky),
            "--target",
            "label",
            "--models",
            "tree",
            "--cv",
            "3",
            "--no-save",
            "--group",
            "customer",
        ],
    )
    assert result.exit_code == 0, result.output


def test_grouping_reaches_tune_and_ensemble(leaky: Path) -> None:
    """All three verbs cross-validate, so all three had the same hole."""
    from michi.cli.app import app

    for argv in (
        [
            "tune",
            str(leaky),
            "--target",
            "label",
            "--model",
            "tree",
            "--candidates",
            "3",
            "--cv",
            "2",
            "--inner-cv",
            "2",
            "--group",
            "customer",
        ],
        [
            "ensemble",
            str(leaky),
            "--target",
            "label",
            "--models",
            "tree,linear",
            "--cv",
            "3",
            "--no-save",
            "--group",
            "customer",
        ],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"{argv[0]}: {result.output}"
