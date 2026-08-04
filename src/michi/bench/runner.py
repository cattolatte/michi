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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from michi.bench.preprocess import PreparationPolicy, build_pipeline, describe_policy
from michi.bench.registry import apply_params, build_model, model_entry
from michi.bench.significance import Comparison, compare_to_leader
from michi.core.artifacts import Finding, Severity
from michi.core.errors import DataError, RunError
from michi.core.io import LoadedTable
from michi.core.manifest import Metric, ModelSpec, RunManifest, capture_environment
from michi.evaluation import detect_task, new_run_id
from michi.recipes.model import Recipe

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

__all__ = [
    "BenchResult",
    "ModelResult",
    "fold_pipeline",
    "make_splitter",
    "run_benchmark",
    "scorers_for",
]

_MIN_ROWS_PER_FOLD = 5


@dataclass(frozen=True, slots=True)
class ModelResult:
    """One model's cross-validated performance."""

    name: str
    metrics: tuple[Metric, ...]
    fold_scores: tuple[float, ...]
    fit_seconds: float
    failed: str | None = None
    oof: tuple[float, ...] = field(default_factory=tuple)
    """Out-of-fold predictions, in the input's row order, when collected.

    Every row is predicted by a fold that did not train on it, which is what
    makes these safe to stack on — and what makes them different from
    predictions of the training data.
    """

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
    recipe: Recipe | None = None,
    seed: int = 0,
    balance: bool = False,
    oof: Path | None = None,
    group: str | None = None,
    metric: str | None = None,
    group_id: str | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
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
    recipe
        A cleaning recipe to apply. Its deterministic steps run once, up
        front; its fitted steps replace michi's default preparation inside
        each fold, because a recipe the user wrote takes precedence over
        michi's assumptions.
    seed
        Seed for fold assignment and every model that accepts one.
    balance
        Weight classes inversely to their frequency, for models that accept
        it. A mechanic, not a judgement: it changes what the loss counts, not
        what michi thinks the right answer is.
    oof
        Where to write out-of-fold predictions, or ``None`` to skip them.
    metric
        Metric to rank, interval, and significance-test by. Defaults to
        michi's headline metric for the task; a name michi does not know is
        looked up among ``michi.metrics`` entry points.
    group
        Column whose rows must stay in one fold. Required whenever rows share
        an entity: without it, cross-validation reports memory as skill.
    params
        Per-model hyperparameter overrides, keyed by catalogue name. A model
        absent from the mapping trains at its defaults, so comparing a tuned
        model against untuned ones is a thing you can state rather than a
        thing that happens by accident.
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

    if recipe is not None:
        from michi.recipes import apply_deterministic

        frame = apply_deterministic(recipe, frame)
        if target not in frame.columns:
            msg = f"the recipe removes the target column {target!r}"
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

    if group is not None and group not in usable.columns:
        msg = f"--group names a column not in the data: {group!r}"
        raise DataError(msg)
    group_values = (
        np.asarray(usable[group].astype("object")) if group is not None else None
    )

    features = usable.drop(columns=[target] + ([group] if group else []))
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

    # Parameters are checked against real estimators before the first fold,
    # for two reasons. A name nobody recognises is a silent no-op — the user
    # reads a leaderboard believing it reflects settings that never applied —
    # and a typo caught after a long benchmark has already wasted the run.
    if params:
        unknown_models = [name for name in params if name not in requested]
        if unknown_models:
            offenders = ", ".join(repr(name) for name in sorted(unknown_models))
            benchmarked = ", ".join(requested)
            msg = (
                f"parameters were given for {offenders}, which "
                f"{'is' if len(unknown_models) == 1 else 'are'} not being "
                f"benchmarked. This benchmark trains: {benchmarked}"
            )
            raise RunError(msg)
        for name, block in params.items():
            apply_params(build_model(name, resolved_task, seed), block)

    splitter, folds = make_splitter(resolved_task, folds, labels, seed, group_values)
    scorers = scorers_for(resolved_task, metric)
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
                recipe=recipe,
                seed=seed,
                groups=group_values,
                balance=balance,
                collect_oof=oof is not None,
                params=(params or {}).get(name),
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
            recipe_path=recipe.source.path if recipe else None,
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


def make_splitter(
    task: str,
    folds: int,
    labels: np.ndarray[Any, Any],
    seed: int,
    groups: np.ndarray[Any, Any] | None = None,
) -> tuple[Any, int]:
    """Build the cross-validation splitter, stratifying where it applies.

    When `groups` is given, every row sharing a group lands in the same fold.
    Without it, a dataset with repeated entities — five rows per customer —
    puts four of a customer's rows in training and one in test, and the score
    that comes back is memory rather than generalisation. `michi split` has
    prevented that at the file level since v1.9; not honouring it here left
    michi contradicting its own advice.
    """
    import numpy as np
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    if groups is not None:
        distinct = len(np.unique(groups))
        if distinct < folds:
            # Fewer groups than folds cannot be split without splitting a
            # group, which is the one thing this exists to prevent.
            folds = max(2, distinct)
        if task == "classification":
            return (
                StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed),
                folds,
            )
        return GroupKFold(n_splits=folds), folds

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


def _custom_scorer(name: str) -> tuple[str, Any, bool] | None:
    """Look up a metric contributed by a plugin, or ``None`` if there is none.

    A competition scores on its own metric, and optimising anything else is
    climbing the wrong hill. michi cannot ship every metric anyone will need,
    so `michi.metrics` entry points let a user supply one — and because it is
    an entry point rather than a path to import, the metric travels with the
    environment the run is reproduced in.
    """
    from importlib.metadata import entry_points

    for entry in entry_points(group="michi.metrics"):
        if entry.name != name:
            continue
        try:
            loaded = entry.load()
        except Exception as err:  # third-party failure boundary
            msg = f"metric plugin {name!r} could not be loaded: {err}"
            raise RunError(msg) from err
        function = getattr(loaded, "score", loaded)
        if not callable(function):
            msg = f"metric plugin {name!r} is not callable and has no `score`"
            raise RunError(msg)
        greater = bool(getattr(loaded, "greater_is_better", True))
        return (name, function, greater)
    return None


def scorers_for(
    task: str, metric: str | None = None
) -> tuple[tuple[str, Any, bool], ...]:
    """Metric functions for a task, headline metric first.

    A named `metric` is promoted to the front, so every ranking, interval, and
    significance test in the run is computed against the thing the user
    actually cares about rather than michi's default.
    """
    import numpy as np
    from sklearn import metrics as skm

    if task == "classification":

        def _f1(truth: Any, prediction: Any) -> float:
            average = "binary" if len(np.unique(truth)) <= 2 else "macro"
            return float(
                skm.f1_score(truth, prediction, average=average, zero_division=0)
            )

        return _promote(
            (
                ("balanced_accuracy", skm.balanced_accuracy_score, True),
                ("accuracy", skm.accuracy_score, True),
                ("f1", _f1, True),
                ("mcc", skm.matthews_corrcoef, True),
            ),
            metric,
        )

    def _rmse(truth: Any, prediction: Any) -> float:
        return float(np.sqrt(skm.mean_squared_error(truth, prediction)))

    def _rmsle(truth: Any, prediction: Any) -> float:
        # Negatives have no logarithm; clipping rather than raising keeps a
        # model that undershoots to -0.001 from failing the whole run.
        return float(
            np.sqrt(
                skm.mean_squared_error(
                    np.log1p(np.clip(truth, 0, None)),
                    np.log1p(np.clip(prediction, 0, None)),
                )
            )
        )

    def _mape(truth: Any, prediction: Any) -> float:
        actual = np.asarray(truth, dtype=float)
        nonzero = actual != 0
        if not nonzero.any():
            return float("nan")
        return float(
            np.mean(
                np.abs(
                    (actual[nonzero] - np.asarray(prediction, dtype=float)[nonzero])
                    / actual[nonzero]
                )
            )
        )

    return _promote(
        (
            ("rmse", _rmse, False),
            ("mae", skm.mean_absolute_error, False),
            ("r2", skm.r2_score, True),
            ("rmsle", _rmsle, False),
            ("mape", _mape, False),
        ),
        metric,
    )


def _promote(
    scorers: tuple[tuple[str, Any, bool], ...], metric: str | None
) -> tuple[tuple[str, Any, bool], ...]:
    """Put the requested metric first, pulling in a plugin if michi lacks it.

    The head of this tuple is what the leaderboard ranks by, what the interval
    describes, and what the significance test compares — so promoting is the
    whole of "optimise the metric I actually care about".
    """
    if not metric:
        return scorers

    for index, entry in enumerate(scorers):
        if entry[0] == metric:
            return (entry, *scorers[:index], *scorers[index + 1 :])

    custom = _custom_scorer(metric)
    if custom is not None:
        return (custom, *scorers)

    known = ", ".join(name for name, _, _ in scorers)
    msg = (
        f"unknown metric {metric!r}. Built in for this task: {known}. "
        "Supply your own through a `michi.metrics` entry point."
    )
    raise RunError(msg)


def _run_one_model(
    *,
    name: str,
    features: pd.DataFrame,
    labels: np.ndarray[Any, Any],
    task: str,
    splitter: Any,
    scorers: tuple[tuple[str, Any, bool], ...],
    policy: PreparationPolicy,
    recipe: Recipe | None,
    seed: int,
    groups: np.ndarray[Any, Any] | None = None,
    balance: bool = False,
    collect_oof: bool = False,
    params: Mapping[str, Any] | None = None,
) -> ModelResult:
    """Cross-validate one model, capturing failure rather than aborting."""
    import numpy as np

    entry = model_entry(name)
    started = time.perf_counter()
    per_metric: dict[str, list[float]] = {metric: [] for metric, _, _ in scorers}
    out_of_fold = np.full(len(features), np.nan, dtype=float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train_index, test_index in splitter.split(features, labels, groups):
                pipeline = fold_pipeline(
                    features=features,
                    estimator=_estimator(
                        name, task, seed, balance=balance, params=params
                    ),
                    policy=policy,
                    recipe=recipe,
                    needs_scaling=entry.needs_scaling,
                )
                pipeline.fit(features.iloc[train_index], labels[train_index])
                predictions = pipeline.predict(features.iloc[test_index])
                if collect_oof:
                    # Row order is preserved by writing into the test index,
                    # so the column joins back onto the original frame.
                    out_of_fold[test_index] = predictions
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
        oof=tuple(out_of_fold) if collect_oof else (),
    )


def _estimator(
    name: str,
    task: str,
    seed: int,
    *,
    balance: bool,
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Build a model, weighting classes by inverse frequency when asked.

    Not every estimator accepts `class_weight`, and one that does not is not
    an error — the request simply does not apply to it, and saying so would
    be noise on a leaderboard where other models did honour it.
    """
    estimator = build_model(name, task, seed)
    if params:
        apply_params(estimator, params)
    if not balance or task != "classification":
        return estimator
    if "class_weight" in getattr(estimator, "get_params", dict)():
        estimator.set_params(class_weight="balanced")
    return estimator


def fold_pipeline(
    *,
    features: pd.DataFrame,
    estimator: Any,
    policy: PreparationPolicy,
    recipe: Recipe | None,
    needs_scaling: bool,
) -> Any:
    """Build the per-fold pipeline, letting the recipe override where it speaks.

    A recipe names specific columns. Those get exactly what it asked for; every
    other column still has to reach the estimator as a number, so michi's
    documented preparation covers the remainder. Passing them through
    untouched would hand the model raw strings and fail the fold.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    if recipe is not None:
        from michi.bench.preprocess import column_specs
        from michi.recipes import transformer_specs

        specs = transformer_specs(recipe, features)
        if specs:
            claimed = {name for _, _, columns in specs for name in columns}
            if needs_scaling:
                # A recipe's fitted step replaces michi's handling of the
                # columns it names — including, until this was fixed, the
                # standardisation a scale-sensitive model depends on. An
                # imputed salary reached the estimator in the tens of
                # thousands while every other feature sat near zero, which
                # silently flattened distance- and gradient-based models.
                # The recipe still decides *what* happens to the column;
                # scaling is appended after it, not instead of it.
                specs = [
                    (name, _scaled(transformer), columns)
                    for name, transformer, columns in specs
                ]
            specs.extend(
                column_specs(
                    features, policy, needs_scaling=needs_scaling, skip=claimed
                )
            )
            preparation = ColumnTransformer(specs, remainder="drop")
            return Pipeline([("prepare", preparation), ("model", estimator)])

    return build_pipeline(features, estimator, policy, needs_scaling=needs_scaling)


def _scaled(transformer: Any) -> Any:
    """Append standardisation to a recipe transformer, preserving its choice."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("recipe", transformer), ("scale", StandardScaler())])


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
                summary="the dummy baseline leads everything tried",
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
    recipe_path: str | None = None,
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
            "recipe": recipe_path,
            "comparison": comparison.to_dict() if comparison else None,
            "fit_seconds": round(result.fit_seconds, 4),
        },
        seed=seed,
        n_rows=n_rows,
        duration_s=result.fit_seconds,
        environment=capture_environment(),
    )
