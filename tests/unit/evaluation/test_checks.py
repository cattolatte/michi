"""Tests for evaluation checks.

These verify that michi catches the mistakes a careful reviewer would ask
about, and stays quiet when there is nothing to say.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from michi.adapters import load_model
from michi.core.artifacts import Severity
from michi.core.io import load_table
from michi.evaluation import evaluate_model


class _Constant:
    """A model that always predicts the same label."""

    def __init__(self, value: int) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        return np.full(features.shape[0], self.value)


class _Oracle:
    """A model that reads the answer from a leaked column."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        return np.asarray(features["leak"])


class _Echo:
    """A model that repeats one feature back as its prediction."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        return np.asarray(features["feature"])


def _kinds(
    model_path: Path, data_path: Path, target: str, **kwargs: object
) -> set[str]:
    manifest = evaluate_model(
        load_model(str(model_path)),
        load_table(data_path),
        target=target,
        bootstrap=0,
        **kwargs,  # type: ignore[arg-type]
    )
    return {check.kind for check in manifest.checks}


def _write_model(tmp_path: Path, model: object, name: str) -> Path:
    path = tmp_path / name
    joblib.dump(model, path)
    return path


# --- baselines -------------------------------------------------------------


def test_flags_a_model_that_loses_to_the_baseline(tmp_path: Path) -> None:
    """A model no better than guessing the mode is reported at high severity."""
    rows = 300
    frame = pd.DataFrame(
        {
            "feature": np.arange(rows) % 7,
            "label": [0] * (rows - 60) + [1] * 60,
        }
    )
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    model_path = _write_model(tmp_path, _Constant(1), "constant.pkl")
    assert "below-baseline" in _kinds(model_path, data_path, "label")


def test_confirms_a_model_that_beats_the_baseline(
    classification_data: tuple[Path, Path],
) -> None:
    """Beating the trivial baselines is recorded, at informational severity."""
    model_path, data_path = classification_data
    assert "beats-baseline" in _kinds(model_path, data_path, "churned")


# --- leakage and degenerate models -----------------------------------------


def test_flags_a_suspiciously_perfect_score(tmp_path: Path) -> None:
    """A perfect score is reported as suspicious rather than celebrated."""
    rows = 300
    label = (np.arange(rows) % 2).astype(int)
    frame = pd.DataFrame({"leak": label, "noise": np.arange(rows) % 5, "label": label})
    data_path = tmp_path / "leaky.csv"
    frame.to_csv(data_path, index=False)
    model_path = _write_model(tmp_path, _Oracle(), "oracle.pkl")
    assert "suspiciously-perfect" in _kinds(model_path, data_path, "label")


def test_flags_single_class_predictions(tmp_path: Path) -> None:
    """A model that only ever predicts one class is reported."""
    rows = 300
    frame = pd.DataFrame(
        {
            "feature": np.arange(rows) % 7,
            "label": [0] * 150 + [1] * 150,
        }
    )
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    model_path = _write_model(tmp_path, _Constant(0), "constant.pkl")
    assert "single-class-predictions" in _kinds(model_path, data_path, "label")


# --- sample size and slices ------------------------------------------------


def test_flags_a_small_evaluation_set(tmp_path: Path) -> None:
    """Few rows is itself reported, because every metric is then uncertain."""
    frame = pd.DataFrame({"feature": range(30), "label": [0, 1] * 15})
    data_path = tmp_path / "small.csv"
    frame.to_csv(data_path, index=False)
    model_path = _write_model(tmp_path, _Constant(1), "constant.pkl")
    assert "small-evaluation-set" in _kinds(model_path, data_path, "label")


def test_flags_uneven_performance_across_slices(tmp_path: Path) -> None:
    """A large subgroup gap is surfaced rather than hidden by the average."""
    rows = 400
    half = rows // 2
    # In group "a" the label always matches the feature; in group "b" it never
    # does. Any model keying on the feature is perfect on one and useless on
    # the other — an aggregate score would hide that completely.
    feature = np.tile([0, 1], rows // 2)
    group = np.array(["a"] * half + ["b"] * half)
    label = np.where(group == "a", feature, 1 - feature)
    frame = pd.DataFrame({"feature": feature, "group": group, "label": label})
    data_path = tmp_path / "slices.csv"
    frame.to_csv(data_path, index=False)

    model_path = _write_model(tmp_path, _Echo(), "echo.pkl")
    manifest = evaluate_model(
        load_model(str(model_path)),
        load_table(data_path),
        target="label",
        slice_columns=("group",),
        bootstrap=0,
    )
    gaps = [check for check in manifest.checks if check.kind == "slice-gap"]
    assert gaps
    assert gaps[0].columns == ("group",)


# --- silence ---------------------------------------------------------------


def test_a_sound_evaluation_raises_no_severe_checks(
    regression_data: tuple[Path, Path],
) -> None:
    """A well-behaved model produces no high-severity checks."""
    model_path, data_path = regression_data
    manifest = evaluate_model(
        load_model(str(model_path)),
        load_table(data_path),
        target="value",
        bootstrap=0,
    )
    assert [c for c in manifest.checks if c.severity is Severity.HIGH] == []
