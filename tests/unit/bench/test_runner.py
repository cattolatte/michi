"""Tests for benchmarking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from michi.bench import PreparationPolicy, available_models, model_entry, run_benchmark
from michi.core.errors import DataError, RunError
from michi.core.io import load_table


def _bench(path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    return run_benchmark(load_table(path), **kwargs)  # type: ignore[arg-type]


# --- the catalogue ---------------------------------------------------------


def test_catalogue_lists_models_for_a_task() -> None:
    """The menu is filtered to what can actually train for the task."""
    names = {entry.name for entry in available_models("regression")}
    assert "lasso" in names
    assert "naive-bayes" not in names


def test_every_entry_describes_itself_without_judging() -> None:
    """Summaries are factual; michi never calls a model best."""
    banned = ("best", "recommended", "superior", "you should")
    for entry in available_models():
        assert entry.summary
        assert not any(word in entry.summary.lower() for word in banned)


def test_unknown_model_lists_what_exists() -> None:
    """A typo is answered with the menu, not a bare failure."""
    with pytest.raises(RunError, match="Available models"):
        model_entry("randomforest")


def test_model_rejects_an_unsupported_task(tidy_csv: Path) -> None:
    """A regression-only model is refused for classification, with reasons."""
    with pytest.raises(RunError, match="does not support"):
        _bench(tidy_csv, target="label", models=("lasso",), task="classification")


# --- running a benchmark ---------------------------------------------------


def test_benchmark_trains_every_requested_model(tidy_csv: Path) -> None:
    """Each requested model appears in the results."""
    result = _bench(tidy_csv, target="label", models=("linear", "tree"), folds=3)
    names = {item.name for item in result.results}
    assert {"linear", "tree"} <= names


def test_dummy_baseline_is_always_added(tidy_csv: Path) -> None:
    """The floor is included whether or not the user asked for it."""
    result = _bench(tidy_csv, target="label", models=("linear",), folds=3)
    assert "dummy" in {item.name for item in result.results}


def test_results_are_ranked_best_first(classification_data: tuple[Path, Path]) -> None:
    """The leader heads the table, by the headline metric."""
    _, data_path = classification_data
    result = _bench(data_path, target="churned", models=("rf", "linear"), folds=3)
    successful = [item for item in result.results if item.failed is None]
    assert successful[0].primary.value >= successful[-1].primary.value


def test_every_model_records_one_score_per_fold(tidy_csv: Path) -> None:
    """Per-fold scores are what the significance test consumes."""
    result = _bench(tidy_csv, target="label", models=("tree",), folds=4)
    tree = next(item for item in result.results if item.name == "tree")
    assert len(tree.fold_scores) == 4


def test_metrics_carry_intervals_across_folds(tidy_csv: Path) -> None:
    """Fold spread is reported, not just the mean."""
    result = _bench(tidy_csv, target="label", models=("tree",), folds=4)
    tree = next(item for item in result.results if item.name == "tree")
    assert tree.primary.has_interval


def test_regression_is_detected_and_ranked_by_rmse(
    regression_data: tuple[Path, Path],
) -> None:
    """A continuous target produces regression metrics, lower being better."""
    _, data_path = regression_data
    result = _bench(data_path, target="value", models=("linear",), folds=3)
    assert result.task == "regression"
    assert result.primary_metric == "rmse"
    assert result.leader is not None
    assert result.leader.primary.greater_is_better is False


def test_benchmark_is_reproducible(classification_data: tuple[Path, Path]) -> None:
    """The same seed produces the same fold scores."""
    _, data_path = classification_data
    first = _bench(data_path, target="churned", models=("tree",), folds=3, seed=7)
    second = _bench(data_path, target="churned", models=("tree",), folds=3, seed=7)
    assert first.results[0].fold_scores == second.results[0].fold_scores


# --- comparisons -----------------------------------------------------------


def test_a_leader_is_identified(classification_data: tuple[Path, Path]) -> None:
    """One model is marked leader, and only one."""
    _, data_path = classification_data
    result = _bench(data_path, target="churned", models=("rf", "linear"), folds=3)
    leaders = [item for item in result.comparisons if item.model == item.leader]
    assert len(leaders) == 1


def test_a_real_model_beats_the_dummy_significantly(tmp_path: Path) -> None:
    """Given a strong signal and enough rows, the floor is distinguishably worse."""
    import numpy as np

    rng = np.random.default_rng(0)
    rows = 1500
    feature = rng.normal(0, 1, rows)
    noise = rng.normal(0, 0.3, rows)
    frame = pd.DataFrame(
        {
            "feature": feature,
            "other": rng.normal(0, 1, rows),
            "label": (feature + noise > 0).astype(int),
        }
    )
    path = tmp_path / "learnable.csv"
    frame.to_csv(path, index=False)

    result = _bench(path, target="label", models=("tree",), folds=5)
    dummy = next(item for item in result.comparisons if item.model == "dummy")
    assert dummy.significant is True


def test_a_modest_gap_on_little_data_is_not_called_significant(
    classification_data: tuple[Path, Path],
) -> None:
    """michi refuses to claim a difference the sample size cannot support.

    This is the correction working: a naive paired t-test over folds would
    call this gap significant.
    """
    _, data_path = classification_data
    result = _bench(data_path, target="churned", models=("rf",), folds=5)
    dummy = next(item for item in result.comparisons if item.model == "dummy")
    assert dummy.difference < 0
    assert dummy.significant is False


def test_ties_are_reported_rather_than_ranked_silently(tidy_csv: Path) -> None:
    """Indistinguishable models are named as such in a check."""
    result = _bench(
        tidy_csv, target="label", models=("linear", "ridge", "tree"), folds=3
    )
    tied = [
        item
        for item in result.comparisons
        if item.model != item.leader and not item.significant
    ]
    if tied:
        assert "no-clear-winner" in {check.kind for check in result.checks}


# --- preparation -----------------------------------------------------------


def test_categorical_columns_are_handled(messy_csv: Path) -> None:
    """A frame full of strings and missing values still trains."""
    result = run_benchmark(
        load_table(messy_csv),
        target="purchased",
        models=("tree",),
        folds=3,
    )
    tree = next(item for item in result.results if item.name == "tree")
    assert tree.failed is None


def test_preparation_policy_is_recorded(tidy_csv: Path) -> None:
    """What was done to the columns is written into every manifest."""
    result = _bench(tidy_csv, target="label", models=("linear",), folds=3)
    assert result.manifests[0].details["preparation"]["numeric_impute"] == "median"


def test_preparation_policy_is_configurable(tidy_csv: Path) -> None:
    """The user can override the defaults; they are not michi's decision."""
    result = _bench(
        tidy_csv,
        target="label",
        models=("linear",),
        folds=3,
        policy=PreparationPolicy(numeric_impute="mean", encode="ordinal"),
    )
    prepared = result.manifests[0].details["preparation"]
    assert prepared["numeric_impute"] == "mean"
    assert prepared["encode"] == "ordinal"


# --- manifests -------------------------------------------------------------


def test_one_manifest_per_successful_model(tidy_csv: Path) -> None:
    """Each model's result is independently recorded and reproducible."""
    result = _bench(tidy_csv, target="label", models=("linear", "tree"), folds=3)
    successful = [item for item in result.results if item.failed is None]
    assert len(result.manifests) == len(successful)


def test_manifests_share_a_group_id(tidy_csv: Path) -> None:
    """Models from one benchmark are linked, so a report can regroup them."""
    result = _bench(tidy_csv, target="label", models=("linear", "tree"), folds=3)
    groups = {item.details["group_id"] for item in result.manifests}
    assert len(groups) == 1


def test_manifests_record_the_comparison_verdict(tidy_csv: Path) -> None:
    """The significance outcome survives into the artifact."""
    result = _bench(tidy_csv, target="label", models=("linear",), folds=3)
    assert result.manifests[0].details["comparison"] is not None


# --- failure modes ---------------------------------------------------------


def test_missing_target_names_available_columns(tidy_csv: Path) -> None:
    """A target typo lists what the data holds."""
    with pytest.raises(DataError, match="available columns"):
        _bench(tidy_csv, target="nope", models=("linear",))


def test_too_few_rows_for_the_folds_is_reported(tmp_path: Path) -> None:
    """Asking for more folds than the data supports fails clearly."""
    path = tmp_path / "small.csv"
    pd.DataFrame({"a": range(10), "label": [0, 1] * 5}).to_csv(path, index=False)
    with pytest.raises(DataError, match="too few"):
        _bench(path, target="label", models=("linear",), folds=5)


def test_one_model_failing_does_not_abort_the_benchmark(messy_csv: Path) -> None:
    """A model that cannot train is recorded as failed; the rest still run."""
    result = run_benchmark(
        load_table(messy_csv), target="purchased", models=("tree", "linear"), folds=3
    )
    assert any(item.failed is None for item in result.results)


def test_folds_shrink_to_what_a_rare_class_supports(tmp_path: Path) -> None:
    """Stratification cannot exceed the rarest class, so michi adapts."""
    rows = 60
    frame = pd.DataFrame(
        {
            "feature": range(rows),
            "label": [0] * (rows - 3) + [1] * 3,
        }
    )
    path = tmp_path / "rare.csv"
    frame.to_csv(path, index=False)
    result = _bench(path, target="label", models=("tree",), folds=5)
    assert result.folds <= 3
