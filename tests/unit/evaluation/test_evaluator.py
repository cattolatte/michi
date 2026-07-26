"""Tests for model evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from michi.adapters import load_model
from michi.core.errors import DataError
from michi.core.io import load_table
from michi.evaluation import detect_task, evaluate_model


def _evaluate(model_path: Path, data_path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    return evaluate_model(
        load_model(str(model_path)),
        load_table(data_path),
        bootstrap=int(kwargs.pop("bootstrap", 50)),
        **kwargs,  # type: ignore[arg-type]
    )


# --- task detection --------------------------------------------------------


def test_detects_classification_from_few_labels() -> None:
    """A binary target is classification, whatever its dtype."""
    assert detect_task(np.array([0, 1, 1, 0])) == "classification"
    assert detect_task(np.array(["yes", "no", "yes"])) == "classification"


def test_detects_regression_from_continuous_values() -> None:
    """A continuous target is regression."""
    assert detect_task(np.linspace(0, 100, 400)) == "regression"


# --- classification --------------------------------------------------------


def test_evaluates_a_classifier(classification_data: tuple[Path, Path]) -> None:
    """A classifier produces classification metrics and a confusion matrix."""
    manifest = _evaluate(*classification_data, target="churned")
    assert manifest.task == "classification"
    assert manifest.metric("balanced_accuracy").value > 0.0
    assert manifest.details["confusion"]


def test_headline_metric_is_balanced_accuracy(
    classification_data: tuple[Path, Path],
) -> None:
    """The first metric is the one that resists class imbalance."""
    manifest = _evaluate(*classification_data, target="churned")
    assert manifest.primary.name == "balanced_accuracy"


def test_classifier_beats_the_dummy_baseline(
    classification_data: tuple[Path, Path],
) -> None:
    """A trained model on learnable data outperforms predicting the mode."""
    manifest = _evaluate(*classification_data, target="churned")
    baseline = next(
        metric
        for metric in manifest.baselines["most_frequent"]
        if metric.name == "balanced_accuracy"
    )
    assert manifest.metric("balanced_accuracy").value > baseline.value


def test_baselines_are_always_recorded(
    classification_data: tuple[Path, Path],
) -> None:
    """Every run answers 'compared to what?' without being asked."""
    manifest = _evaluate(*classification_data, target="churned")
    assert "most_frequent" in manifest.baselines
    assert "stratified" in manifest.baselines


# --- regression ------------------------------------------------------------


def test_evaluates_a_regressor(regression_data: tuple[Path, Path]) -> None:
    """A regressor produces regression metrics, headed by RMSE."""
    manifest = _evaluate(*regression_data, target="value")
    assert manifest.task == "regression"
    assert manifest.primary.name == "rmse"
    assert manifest.metric("r2").value > 0.5


def test_error_metrics_are_marked_lower_is_better(
    regression_data: tuple[Path, Path],
) -> None:
    """Metric direction is recorded so comparisons never guess."""
    manifest = _evaluate(*regression_data, target="value")
    assert manifest.metric("rmse").greater_is_better is False
    assert manifest.metric("r2").greater_is_better is True


# --- uncertainty -----------------------------------------------------------


def test_metrics_carry_confidence_intervals(
    classification_data: tuple[Path, Path],
) -> None:
    """Headline metrics report an interval, not just a point estimate."""
    manifest = _evaluate(*classification_data, target="churned", bootstrap=100)
    metric = manifest.metric("accuracy")
    assert metric.has_interval
    assert metric.ci_low <= metric.value <= metric.ci_high  # type: ignore[operator]


def test_bootstrap_can_be_disabled(classification_data: tuple[Path, Path]) -> None:
    """Intervals are skippable when speed matters more than uncertainty."""
    manifest = _evaluate(*classification_data, target="churned", bootstrap=0)
    assert manifest.metric("accuracy").has_interval is False


def test_intervals_are_reproducible(classification_data: tuple[Path, Path]) -> None:
    """The same seed yields the same interval."""
    first = _evaluate(*classification_data, target="churned", bootstrap=100, seed=5)
    second = _evaluate(*classification_data, target="churned", bootstrap=100, seed=5)
    assert first.metric("accuracy").ci_low == second.metric("accuracy").ci_low


# --- slices and calibration ------------------------------------------------


def test_slices_are_computed_automatically(
    classification_data: tuple[Path, Path],
) -> None:
    """Low-cardinality columns are scored separately without being asked."""
    manifest = _evaluate(*classification_data, target="churned")
    columns = {item["column"] for item in manifest.details["slices"]}
    assert "region" in columns


def test_explicit_slice_columns_are_honoured(
    classification_data: tuple[Path, Path],
) -> None:
    """Naming slice columns restricts the analysis to them."""
    manifest = _evaluate(
        *classification_data, target="churned", slice_columns=("region",)
    )
    assert {item["column"] for item in manifest.details["slices"]} == {"region"}


def test_unknown_slice_column_is_reported(
    classification_data: tuple[Path, Path],
) -> None:
    """A slice column that does not exist fails clearly."""
    with pytest.raises(DataError, match="slice column"):
        _evaluate(*classification_data, target="churned", slice_columns=("absent",))


def test_calibration_is_recorded_for_probability_models(
    classification_data: tuple[Path, Path],
) -> None:
    """A model with probabilities gets a calibration curve and an error score."""
    manifest = _evaluate(*classification_data, target="churned")
    assert manifest.details["calibration"]
    assert manifest.details["ece"] >= 0.0


# --- provenance ------------------------------------------------------------


def test_manifest_records_provenance(
    classification_data: tuple[Path, Path],
) -> None:
    """A result is traceable to the bytes, model, seed, and environment."""
    manifest = _evaluate(*classification_data, target="churned", seed=3)
    assert len(manifest.dataset.sha256) == 64
    assert manifest.model.sha256 is not None
    assert manifest.seed == 3
    assert manifest.environment.packages["michi"]


def test_run_ids_are_unique() -> None:
    """Two runs never collide, so manifests never overwrite each other."""
    from michi.evaluation import new_run_id

    assert new_run_id() != new_run_id()


# --- failure modes ---------------------------------------------------------


def test_missing_target_names_available_columns(
    classification_data: tuple[Path, Path],
) -> None:
    """A target typo lists what the data actually holds."""
    with pytest.raises(DataError, match="available columns"):
        _evaluate(*classification_data, target="not_a_column")


def test_all_missing_target_is_reported(
    classification_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """A target column with no values cannot be evaluated against."""
    model_path, data_path = classification_data
    frame = pd.read_csv(data_path)
    frame["churned"] = None
    empty = tmp_path / "empty_target.csv"
    frame.to_csv(empty, index=False)
    with pytest.raises(DataError, match="missing"):
        _evaluate(model_path, empty, target="churned")


def test_rows_with_missing_labels_are_skipped(
    classification_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Unlabelled rows cannot be scored, so they are excluded and counted."""
    model_path, data_path = classification_data
    frame = pd.read_csv(data_path)
    frame.loc[frame.index[:10], "churned"] = None
    partial = tmp_path / "partial.csv"
    frame.to_csv(partial, index=False)
    manifest = _evaluate(model_path, partial, target="churned")
    assert manifest.n_rows == len(frame) - 10
