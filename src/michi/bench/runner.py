"""Training and comparing several models under cross-validation.

Design Principles
-----------------
- **The comparison is the product.** Training many models is easy; saying
  honestly which of them is actually better is not, and that is what this
  module exists to do.
- **A dummy baseline is always included.** It is added to every benchmark
  whether or not the user asked for it, because a leaderboard without a floor
  invites the wrong conclusion.
- **Everything happens inside folds.** Preparation is fitted per fold, so no
  benchmark can leak test information into training through an imputer or an
  encoder.
- **michi ranks, but never chooses.** Results are ordered and differences are
  tested; picking the model remains the user's decision, and a leader that is
  statistically indistinguishable from the rest is reported as such.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from michi.bench.preprocess import PreparationPolicy, build_pipeline, describe_policy
from michi.bench.registry import build_model, model_entry
from michi.bench.significance import Comparison, compare_to_leader
from michi.core.artifacts import Finding, Severity
from michi.core.errors import DataError, RunError
from michi.core.io import LoadedTable
from michi.core.manifest import Metric, ModelSpec, RunManifest, capture_environment
from michi.evaluation import detect_task, new_run_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

__all__ = ["BenchResult", "ModelResult", "run_benchmark"]

_MIN_ROWS_PER_FOLD = 5


@dataclass(frozen=True, slots=True)
class ModelResult:
    """One model's cross-validated performance."""

    name: str
    metrics: tuple[Metric, ...]
    fold_scores: tuple[float, ...]
    fit_seconds: float
    failed: str | None = None

    @property
    def primary(self) -> Metric:
        """The metric the comparison is based on."""
        return self.metrics[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in a run manifest."""
        return {
            "name": self.name,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "fold_scores": list(self.fold_scores),
            "fit_seconds": round(self.fit_seconds, 4),
            "failed": self.failed,
        }


@dataclass(frozen=True, slots=True)
class BenchResult:
    """The outcome of one benchmark: every model, ranked and compared."""

    task: str
    target: str
    folds: int
    primary_metric: str
    results: tuple[ModelResult, ...]
    comparisons: tuple[Comparison, ...]
    policy: PreparationPolicy
    checks: tuple[Finding, ...] = ()
    n_rows: int = 0
    seed: int = 0
    duration_s: float = 0.0
    manifests: tuple[RunManifest, ...] = field(default_factory=tuple)

    @property
    def leader(self) -> ModelResult | None:
        """The best-scoring model, or ``None`` if every model failed."""
        successful = [item for item in self.results if item.failed is None]
        return successful[0] if successful else None


def run_benchmark(
    table: LoadedTable,
    *,
    target: str,
    models: tuple[str, ...],
    task: str | None = None,
    folds: int = 5,
    policy: PreparationPolicy | None = None,
    seed: int = 0,
    group_id: str | None = None,
) -> BenchResult:
    """Train and compare models under cross-validation.

    Parameters
    ----------
    table
        The dataset, including its provenance.
    target
        Name of the label column.
    models
        Catalogue names to train. A dummy baseline is always added.
    task
        ``"classification"`` or ``"regression"``; inferred when omitted.
    folds
        Number of cross-validation folds.
    policy
        Column preparation choices; documented defaults are used when omitted.
    seed
        Seed for fold assignment and every model that accepts one.
    group_id
        Identifier shared by all manifests from this benchmark.

    Returns
    -------
    BenchResult
        Ranked results, significance comparisons, and one manifest per model.

    Raises
    ------
    DataError
        If the target is missing, or there are too few rows for the folds.
    RunError
        If no model could be trained at all.
    """
    import numpy as np

    started = time.perf_counter()
    frame = table.frame

    if target not in frame.columns:
        available = ", ".join(str(name) for name in frame.columns[:10])
        msg = f"target {target!r} is not a column; available columns: {available}"
        raise DataError(msg)

    usable = frame[frame[target].notna()]
    if usable.empty:
        msg = f"every value of target {target!r} is missing"
        raise DataError(msg)
    if usable.shape[0] < folds * _MIN_ROWS_PER_FOLD:
        msg = (
            f"{usable.shape[0]} rows is too few for {folds}-fold "
            f"cross-validation; use --cv with a smaller number"
        )
        raise DataError(msg)

    features = usable.drop(columns=[target])
    labels = np.asarray(usable[target])
    resolved_task = task or detect_task(labels)
    resolved_policy = policy or PreparationPolicy()
    requested = _with_baseline(models)

    # An unknown or inapplicable model name is the user's typo, not a training
    # failure, so it is caught before any fitting starts rather than being
    # reported forty seconds later as one row of a leaderboard.
    for name in requested:
        entry = model_entry(name)
        if not entry.supports(resolved_task):
            supported = ", ".join(sorted(entry.tasks))
            msg = f"{name!r} does not support {resolved_task}; it supports: {supported}"
            raise RunError(msg)

    splitter, folds = _make_splitter(resolved_task, folds, labels, seed)
    scorers = _scorers(resolved_task)
    primary_metric = scorers[0][0]

    results: list[ModelResult] = []
    for name in requested:
        results.append(
            _run_one_model(
                name=name,
                features=features,
                labels=labels,
                task=resolved_task,
                splitter=splitter,
                scorers=scorers,
                policy=resolved_policy,
                seed=seed,
            )
        )

    successful = [item for item in results if item.failed is None]
    if not successful:
        failures = "; ".join(
            f"{item.name}: {item.failed}" for item in results if item.failed
        )
        msg = f"no model could be trained. {failures}"
        raise RunError(msg)

    greater_is_better = successful[0].primary.greater_is_better
    comparisons = compare_to_leader(
        {item.name: np.asarray(item.fold_scores) for item in successful},
        greater_is_better=greater_is_better,
        train_size=int(usable.shape[0] * (folds - 1) / folds),
        test_size=int(usable.shape[0] / folds),
    )
    order = {comparison.model: index for index, comparison in enumerate(comparisons)}
    results.sort(key=lambda item: order.get(item.name, len(order) + 1))

    checks = _benchmark_checks(results, comparisons)
    duration = time.perf_counter() - started
    identifier = group_id or new_run_id(seed)
    manifests = tuple(
        _manifest_for(
            result=result,
            table=table,
            target=target,
            task=resolved_task,
            folds=folds,
            seed=seed,
            policy=resolved_policy,
            comparisons=comparisons,
            group_id=identifier,
            n_rows=int(usable.shape[0]),
        )
        for result in results
        if result.failed is None
    )

    return BenchResult(
        task=resolved_task,
        target=target,
        folds=folds,
        primary_metric=primary_metric,
        results=tuple(results),
        comparisons=comparisons,
        policy=resolved_policy,
        checks=checks,
        n_rows=int(usable.shape[0]),
        seed=seed,
        duration_s=duration,
        manifests=manifests,
    )


def _with_baseline(models: tuple[str, ...]) -> tuple[str, ...]:
    """Always include the dummy floor, however the user phrased the request."""
    names = list(dict.fromkeys(models))
    if "dummy" not in names:
        names.append("dummy")
    return tuple(names)


def _make_splitter(
    task: str, folds: int, labels: np.ndarray[Any, Any], seed: int
) -> tuple[Any, int]:
    """Build the cross-validation splitter, stratifying where it applies."""
    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold

    if task != "classification":
        return KFold(n_splits=folds, shuffle=True, random_state=seed), folds

    _, counts = np.unique(labels, return_counts=True)
    smallest = int(counts.min())
    if smallest < folds:
        # Stratification cannot produce more folds than the rarest class has
        # members; silently dropping stratification would be worse than
        # using the number of folds the data can actually support.
        folds = max(2, smallest)
    return (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        folds,
    )


def _scorers(task: str) -> tuple[tuple[str, Any, bool], ...]:
    """Metric functions for a task, headline metric first."""
    import numpy as np
    from sklearn import metrics as skm

    if task == "classification":

        def _f1(truth: Any, prediction: Any) -> float:
            average = "binary" if len(np.unique(truth)) <= 2 else "macro"
            return float(
                skm.f1_score(truth, prediction, average=average, zero_division=0)
            )

        return (
            ("balanced_accuracy", skm.balanced_accuracy_score, True),
            ("accuracy", skm.accuracy_score, True),
            ("f1", _f1, True),
            ("mcc", skm.matthews_corrcoef, True),
        )

    def _rmse(truth: Any, prediction: Any) -> float:
        return float(np.sqrt(skm.mean_squared_error(truth, prediction)))

    return (
        ("rmse", _rmse, False),
        ("mae", skm.mean_absolute_error, False),
        ("r2", skm.r2_score, True),
    )


def _run_one_model(
    *,
    name: str,
    features: pd.DataFrame,
    labels: np.ndarray[Any, Any],
    task: str,
    splitter: Any,
    scorers: tuple[tuple[str, Any, bool], ...],
    policy: PreparationPolicy,
    seed: int,
) -> ModelResult:
    """Cross-validate one model, capturing failure rather than aborting."""
    import numpy as np

    entry = model_entry(name)
    started = time.perf_counter()
    per_metric: dict[str, list[float]] = {metric: [] for metric, _, _ in scorers}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train_index, test_index in splitter.split(features, labels):
                pipeline = build_pipeline(
                    features,
                    build_model(name, task, seed),
                    policy,
                    needs_scaling=entry.needs_scaling,
                )
                pipeline.fit(features.iloc[train_index], labels[train_index])
                predictions = pipeline.predict(features.iloc[test_index])
                for metric_name, scorer, _ in scorers:
                    per_metric[metric_name].append(
                        float(scorer(labels[test_index], predictions))
                    )
    except Exception as err:  # third-party failure boundary
        return ModelResult(
            name=name,
            metrics=(),
            fold_scores=(),
            fit_seconds=time.perf_counter() - started,
            failed=str(err).splitlines()[0][:160],
        )

    metrics: list[Metric] = []
    for metric_name, _, greater_is_better in scorers:
        values = np.asarray(per_metric[metric_name], dtype=float)
        low, high = _fold_interval(values)
        metrics.append(
            Metric(
                name=metric_name,
                value=float(np.mean(values)),
                ci_low=low,
                ci_high=high,
                greater_is_better=greater_is_better,
            )
        )

    return ModelResult(
        name=name,
        metrics=tuple(metrics),
        fold_scores=tuple(per_metric[scorers[0][0]]),
        fit_seconds=time.perf_counter() - started,
    )


def _fold_interval(values: np.ndarray[Any, Any]) -> tuple[float | None, float | None]:
    """A t-interval for the mean across folds.

    Folds are correlated, so this interval is optimistic — it describes the
    spread of fold scores, not the true generalisation error. The significance
    tests, which correct for that correlation, are what comparisons rely on.
    """
    import numpy as np
    from scipy import stats

    if values.shape[0] < 2:
        return None, None
    mean = float(np.mean(values))
    error = float(stats.sem(values))
    if error == 0.0 or not np.isfinite(error):
        return mean, mean
    low, high = stats.t.interval(0.95, df=values.shape[0] - 1, loc=mean, scale=error)
    return float(low), float(high)


def _benchmark_checks(
    results: list[ModelResult], comparisons: tuple[Comparison, ...]
) -> tuple[Finding, ...]:
    """Raise what a reviewer would ask about a leaderboard."""
    findings: list[Finding] = []
    successful = [item for item in results if item.failed is None]
    if not successful:
        return ()

    leader = successful[0]
    dummy = next((item for item in successful if item.name == "dummy"), None)

    if dummy is not None and leader.name != "dummy":
        better = (
            leader.primary.value > dummy.primary.value
            if leader.primary.greater_is_better
            else leader.primary.value < dummy.primary.value
        )
        if not better:
            findings.append(
                Finding(
                    kind="below-baseline",
                    severity=Severity.HIGH,
                    columns=(),
                    summary=(
                        f"no model beats the dummy baseline "
                        f"({leader.primary.name} {leader.primary.value:.4g} vs "
                        f"{dummy.primary.value:.4g})"
                    ),
                    metrics={"leader": leader.name},
                )
            )
    elif dummy is not None and leader.name == "dummy":
        findings.append(
            Finding(
                kind="below-baseline",
                severity=Severity.HIGH,
                columns=(),
                summary="the dummy baseline scores highest of everything tried",
                metrics={"leader": "dummy"},
            )
        )

    tied = [
        comparison
        for comparison in comparisons
        if comparison.model != comparison.leader and not comparison.significant
    ]
    if tied:
        names = ", ".join(comparison.model for comparison in tied[:4])
        findings.append(
            Finding(
                kind="no-clear-winner",
                severity=Severity.WARN,
                columns=(),
                summary=(
                    f"{len(tied)} model(s) are not statistically distinguishable "
                    f"from {comparisons[0].model}: {names}"
                ),
                metrics={"tied": len(tied)},
            )
        )

    failed = [item for item in results if item.failed is not None]
    for item in failed:
        findings.append(
            Finding(
                kind="model-failed",
                severity=Severity.WARN,
                columns=(),
                summary=f"{item.name} could not be trained: {item.failed}",
                metrics={"model": item.name},
            )
        )

    return tuple(sorted(findings, key=lambda f: (f.severity.rank, f.kind)))


def _manifest_for(
    *,
    result: ModelResult,
    table: LoadedTable,
    target: str,
    task: str,
    folds: int,
    seed: int,
    policy: PreparationPolicy,
    comparisons: tuple[Comparison, ...],
    group_id: str,
    n_rows: int,
) -> RunManifest:
    """Record one model's benchmark result as a durable manifest."""
    comparison = next((item for item in comparisons if item.model == result.name), None)
    return RunManifest(
        run_id=f"{group_id}-{result.name}",
        kind="bench",
        dataset=table.source,
        target=target,
        task=task,
        model=ModelSpec(
            reference=result.name,
            loader="registry",
            class_name=result.name,
        ),
        metrics=result.metrics,
        details={
            "group_id": group_id,
            "folds": folds,
            "fold_scores": list(result.fold_scores),
            "preparation": policy.to_dict(),
            "preparation_summary": describe_policy(policy, scaled=True),
            "comparison": comparison.to_dict() if comparison else None,
            "fit_seconds": round(result.fit_seconds, 4),
        },
        seed=seed,
        n_rows=n_rows,
        duration_s=result.fit_seconds,
        environment=capture_environment(),
    )
