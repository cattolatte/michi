"""Tests for `michi split` and permutation importance.

Both exist because a default lies. A random split lies when rows share an
entity or the data is a time series; an unmeasured feature list lies by
letting a reader assume the model uses what they expect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from michi.cli.app import app
from michi.evaluation.importance import ColumnImportance, permutation_importance

runner = CliRunner()


@pytest.fixture
def grouped(tmp_path: Path) -> Path:
    """Rows that belong to entities — five per customer."""
    rng = np.random.default_rng(0)
    rows = 300
    signal = rng.normal(size=rows)
    pd.DataFrame(
        {
            "customer": [f"c{i // 5}" for i in range(rows)],
            "when": pd.date_range("2022-01-01", periods=rows, freq="D").astype(str),
            "signal": signal,
            "label": (signal + rng.normal(scale=0.3, size=rows) > 0).astype(int),
        }
    ).to_csv(tmp_path / "d.csv", index=False)
    return tmp_path / "d.csv"


# --- split -----------------------------------------------------------------


def test_a_grouped_split_keeps_an_entity_on_one_side(
    grouped: Path, tmp_path: Path
) -> None:
    """This is the whole point: the same customer must not span both sides.

    A random split puts four of a customer's rows in training and one in test,
    and the score that comes back is memory rather than generalisation.
    """
    result = runner.invoke(
        app,
        [
            "split",
            str(grouped),
            "--target",
            "label",
            "--group",
            "customer",
            "--train",
            str(tmp_path / "tr.csv"),
            "--test",
            str(tmp_path / "te.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    train = pd.read_csv(tmp_path / "tr.csv")
    test = pd.read_csv(tmp_path / "te.csv")
    assert not set(train["customer"]) & set(test["customer"])


def test_a_time_split_holds_out_the_future(grouped: Path, tmp_path: Path) -> None:
    """A model must never be asked to predict the past."""
    runner.invoke(
        app,
        [
            "split",
            str(grouped),
            "--time",
            "when",
            "--train",
            str(tmp_path / "tr.csv"),
            "--test",
            str(tmp_path / "te.csv"),
        ],
    )
    train = pd.read_csv(tmp_path / "tr.csv")
    test = pd.read_csv(tmp_path / "te.csv")
    assert train["when"].max() <= test["when"].min()


def test_a_categorical_target_is_stratified_without_being_asked(
    grouped: Path, tmp_path: Path
) -> None:
    """An unstratified split on an imbalanced target can starve a fold."""
    result = runner.invoke(
        app,
        [
            "split",
            str(grouped),
            "--target",
            "label",
            "--train",
            str(tmp_path / "tr.csv"),
            "--test",
            str(tmp_path / "te.csv"),
        ],
    )
    assert "stratified" in result.output
    train = pd.read_csv(tmp_path / "tr.csv")
    test = pd.read_csv(tmp_path / "te.csv")
    assert abs(train["label"].mean() - test["label"].mean()) < 0.06


def test_an_explicit_group_outranks_the_balance_heuristic(
    grouped: Path, tmp_path: Path
) -> None:
    """A named column is a constraint michi could not have inferred."""
    result = runner.invoke(
        app,
        [
            "split",
            str(grouped),
            "--target",
            "label",
            "--group",
            "customer",
            "--train",
            str(tmp_path / "tr.csv"),
            "--test",
            str(tmp_path / "te.csv"),
        ],
    )
    assert "group" in result.output
    assert "stratified" not in result.output


def test_a_random_split_says_what_it_cannot_promise(
    grouped: Path, tmp_path: Path
) -> None:
    """The default that lies should say when it might be lying."""
    result = runner.invoke(
        app,
        [
            "split",
            str(grouped),
            "--strategy",
            "random",
            "--train",
            str(tmp_path / "tr.csv"),
            "--test",
            str(tmp_path / "te.csv"),
        ],
    )
    combined = " ".join(result.output.split())
    assert "--group" in combined and "--time" in combined


def test_an_impossible_test_size_is_refused(grouped: Path) -> None:
    """Holding out everything is not a split."""
    result = runner.invoke(app, ["split", str(grouped), "--test-size", "1.5"])
    assert result.exit_code == 2


def test_the_input_file_is_never_modified(grouped: Path, tmp_path: Path) -> None:
    """michi does not write over the data it was given."""
    before = grouped.read_bytes()
    runner.invoke(
        app,
        [
            "split",
            str(grouped),
            "--strategy",
            "random",
            "--train",
            str(tmp_path / "tr.csv"),
            "--test",
            str(tmp_path / "te.csv"),
        ],
    )
    assert grouped.read_bytes() == before


# --- importance ------------------------------------------------------------


class _Model:
    """A model that uses one column and ignores the other."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame["signal"] > 0).astype(int).to_numpy()


def test_importance_finds_the_column_the_model_uses() -> None:
    """Shuffling the column a model reads should cost it."""
    rng = np.random.default_rng(0)
    rows = 200
    signal = rng.normal(size=rows)
    frame = pd.DataFrame({"signal": signal, "ignored": rng.normal(size=rows)})
    labels = (signal > 0).astype(int)

    from sklearn.metrics import balanced_accuracy_score

    ranked = permutation_importance(
        _Model(), frame, labels, scorer=balanced_accuracy_score, seed=0
    )
    assert ranked[0].column == "signal"
    assert ranked[0].drop > 0.2


def test_a_column_the_model_ignores_measures_as_noise() -> None:
    """An importance inside its own error bar is not a finding."""
    rng = np.random.default_rng(0)
    rows = 200
    signal = rng.normal(size=rows)
    frame = pd.DataFrame({"signal": signal, "ignored": rng.normal(size=rows)})
    labels = (signal > 0).astype(int)

    from sklearn.metrics import balanced_accuracy_score

    ranked = permutation_importance(
        _Model(), frame, labels, scorer=balanced_accuracy_score, seed=0
    )
    ignored = next(item for item in ranked if item.column == "ignored")
    assert ignored.is_noise


def test_noise_is_judged_against_the_spread_not_a_constant() -> None:
    """A big drop with a bigger spread is still not a finding."""
    assert ColumnImportance("a", drop=0.2, spread=0.01).is_noise is False
    assert ColumnImportance("b", drop=0.2, spread=0.30).is_noise is True


def test_a_frame_too_wide_to_measure_is_declined() -> None:
    """Importance on hundreds of columns costs more than it tells anyone."""
    frame = pd.DataFrame({f"c{i}": [1.0, 2.0] for i in range(200)})
    assert (
        permutation_importance(
            _Model(), frame, np.array([0, 1]), scorer=lambda a, b: 0.0
        )
        == ()
    )


def test_importance_reaches_the_manifest_and_the_terminal(
    grouped: Path, tmp_path: Path
) -> None:
    """The flag has to actually arrive — it once did not."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    frame = pd.read_csv(grouped)
    model = DecisionTreeClassifier(random_state=0).fit(
        frame[["signal"]], frame["label"]
    )
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    frame[["signal", "label"]].to_csv(tmp_path / "eval.csv", index=False)

    result = runner.invoke(
        app,
        [
            "eval",
            str(path),
            str(tmp_path / "eval.csv"),
            "--target",
            "label",
            "--importance",
            "--no-save",
            "--bootstrap",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Column importance" in result.output
