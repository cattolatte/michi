"""Hyperparameter search over a model michi already knows how to build.

Design Principles
-----------------
- **The space is a menu, not a secret.** Every search space is written down
  here, printable with ``--list-space``, and overridable with a YAML file. A
  tuner whose space you cannot see is a tuner making your modelling decisions
  for you and calling it optimisation.
- **The search is nested, or it is a lie.** Hyperparameters are chosen by
  cross-validation *inside* the training folds, and the score michi reports
  comes from a fold the search never touched. Reporting the best inner score
  as if it were performance is the second most common silent leak in tabular
  ML, after target encoding.
- **Search is a tool, not a verdict.** ``tune`` reports what it found and how
  much of the difference survives the noise; it never says the tuned model is
  the one to ship.
- **Deterministic.** Same seed, same space, same result — the whole toolbox
  rests on that, so every sampler is seeded explicitly.
- **A stronger optimiser needs the guardrail more, not less.** A model-based
  search is better at overfitting the inner folds — that is the mechanism
  working, not failing — so the gap between its own best score and the
  held-out one grows with its sophistication. michi reports both for every
  strategy (ADR-0004).
- Strategies come from scikit-learn and Optuna. michi adds the space, the
  nesting, and the honesty about what the number means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from michi.core.errors import RunError, install_hint

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "STRATEGIES",
    "TuneResult",
    "load_space",
    "search_space",
    "tune_model",
]

STRATEGIES: tuple[str, ...] = ("random", "halving", "grid", "bayes")
"""Search strategies michi exposes, in the order the docs describe them."""


# Spaces are deliberately modest. A space so large that the search cannot
# cover it turns tuning into a lottery whose result does not replicate, and
# every one of these is small enough to finish on a laptop.
_SPACES: dict[str, dict[str, list[Any]]] = {
    "ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "linear": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "tree": {
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [1, 5, 20],
    },
    "rf": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [8, 16, None],
        "model__min_samples_leaf": [1, 5, 20],
        "model__max_features": ["sqrt", 0.5, 1.0],
    },
    "extra-trees": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [8, 16, None],
        "model__min_samples_leaf": [1, 5, 20],
    },
    "hist-gbm": {
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 6, None],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 1.0],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 11, 25],
        "model__weights": ["uniform", "distance"],
    },
    "svm": {
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", "auto"],
    },
    "xgb": {
        "model__n_estimators": [200, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 6, 9],
        "model__subsample": [0.7, 1.0],
        "model__colsample_bytree": [0.7, 1.0],
    },
    "lgbm": {
        "model__n_estimators": [200, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
        "model__subsample": [0.7, 1.0],
    },
    "mlp": {
        "model__hidden_layer_sizes": [(64,), (128, 64), (256, 128), (256, 128, 64)],
        "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "model__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
    },
    "torch-mlp": {
        "model__hidden": [(128,), (256, 128), (512, 256, 128)],
        "model__dropout": [0.0, 0.1, 0.3],
        "model__learning_rate": [1e-4, 5e-4, 1e-3, 3e-3],
        "model__batch_size": [64, 256, 1024],
    },
    "catboost": {
        "model__iterations": [200, 500],
        "model__learning_rate": [0.03, 0.1],
        "model__depth": [4, 6, 8],
    },
}


@dataclass(frozen=True, slots=True)
class TuneResult:
    """What a search found, and what it is worth.

    Attributes
    ----------
    model
        Catalogue name of the model searched.
    strategy
        Which search strategy ran.
    best_params
        The winning hyperparameters, with michi's pipeline prefix stripped.
    candidates
        How many configurations were evaluated.
    inner_score
        Best score from the inner search. Optimistic by construction: it is
        the maximum over many draws on the folds that chose it.
    outer_score
        Score on folds the search never saw. This is the honest number.
    baseline_score
        The same model at its defaults, scored the same way.
    metric
        Name of the score.
    greater_is_better
        Direction of improvement for `metric`.
    seconds
        Wall-clock time of the whole search.
    """

    model: str
    strategy: str
    best_params: dict[str, Any]
    candidates: int
    inner_score: float
    outer_score: float
    baseline_score: float
    metric: str
    greater_is_better: bool = True
    seconds: float = 0.0
    fold_scores: tuple[float, ...] = field(default_factory=tuple)
    baseline_folds: tuple[float, ...] = field(default_factory=tuple)

    @property
    def improvement(self) -> float:
        """How much tuning gained over the defaults, signed so up is good."""
        delta = self.outer_score - self.baseline_score
        return delta if self.greater_is_better else -delta

    @property
    def optimism(self) -> float:
        """How much the inner score overstated the honest one.

        The gap between "best of many tries on the data that picked it" and
        "score on data the search never saw" is the number that keeps a tuning
        run honest, and it is almost never zero.
        """
        delta = self.inner_score - self.outer_score
        return delta if self.greater_is_better else -delta


def search_space(model: str) -> dict[str, list[Any]]:
    """The documented search space for one model.

    Raises
    ------
    RunError
        If michi has no space for that model, naming the ones it has.

    Examples
    --------
    >>> "model__alpha" in search_space("ridge")
    True
    """
    if model not in _SPACES:
        known = ", ".join(sorted(_SPACES))
        msg = (
            f"no built-in search space for {model!r}. "
            f"Models with a space: {known}. "
            "Supply your own with --space my_space.yaml."
        )
        raise RunError(msg)
    return {key: list(values) for key, values in _SPACES[model].items()}


def load_space(path: Any) -> dict[str, list[Any]]:
    """Read a user-supplied search space from YAML.

    The file maps a parameter name to the list of values to try. Names may be
    written bare (``max_depth``) or with michi's pipeline prefix
    (``model__max_depth``); the prefix is added when it is missing, because
    requiring users to know michi's internal step names would be a trap.
    """
    from pathlib import Path

    import yaml

    source = Path(path)
    if not source.exists():
        msg = f"no such search space file: {source}"
        raise RunError(msg)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as err:
        msg = f"could not parse {source.name} as YAML: {err}"
        raise RunError(msg) from err
    if not isinstance(payload, dict):
        msg = f"{source.name} should map parameter names to lists of values"
        raise RunError(msg)

    space: dict[str, list[Any]] = {}
    for key, values in payload.items():
        name = str(key) if str(key).startswith("model__") else f"model__{key}"
        if not isinstance(values, list):
            msg = f"{source.name}: {key!r} should be a list of values to try"
            raise RunError(msg)
        space[name] = list(values)
    if not space:
        msg = f"{source.name} is empty — nothing to search"
        raise RunError(msg)
    return space


def tune_model(
    features: pd.DataFrame,
    labels: Any,
    *,
    model: str,
    task: str,
    space: dict[str, list[Any]],
    strategy: str = "random",
    candidates: int = 30,
    folds: int = 5,
    inner_folds: int = 3,
    seed: int = 0,
    policy: Any = None,
    recipe: Any = None,
    groups: Any = None,
    metric: str | None = None,
) -> TuneResult:
    """Search hyperparameters, and score the winner on untouched folds.

    The search runs inside each outer training fold, so the reported
    `outer_score` comes from data no configuration was chosen on. The
    difference between that and `inner_score` is reported rather than hidden,
    because it is the whole reason nested validation exists.
    """
    import time

    import numpy as np

    from michi.bench.preprocess import PreparationPolicy
    from michi.bench.registry import build_model, model_entry
    from michi.bench.runner import _fold_pipeline, _make_splitter, _scorers

    if strategy not in STRATEGIES:
        known = ", ".join(STRATEGIES)
        msg = f"unknown strategy {strategy!r}; michi offers: {known}"
        raise RunError(msg)

    entry = model_entry(model)
    resolved_policy = policy if policy is not None else PreparationPolicy()
    scorers = _scorers(task, metric)
    metric_name, scorer, greater_is_better = scorers[0]

    started = time.perf_counter()
    label_array = np.asarray(labels)
    splitter, folds = _make_splitter(task, folds, label_array, seed, groups)
    outer: list[float] = []
    baseline: list[float] = []
    inner: list[float] = []
    winners: list[dict[str, Any]] = []
    evaluated = 0
    for train_index, test_index in splitter.split(features, label_array, groups):
        train_x = features.iloc[train_index]
        train_y = label_array[train_index]
        test_x = features.iloc[test_index]
        test_y = label_array[test_index]

        pipeline = _fold_pipeline(
            features=train_x,
            estimator=build_model(model, task, seed),
            policy=resolved_policy,
            recipe=recipe,
            needs_scaling=entry.needs_scaling,
        )
        search = _build_search(
            pipeline,
            space,
            strategy=strategy,
            candidates=candidates,
            folds=inner_folds,
            seed=seed,
            task=task,
            labels=train_y,
            groups=None if groups is None else groups[train_index],
            metric=metric_name,
            greater_is_better=greater_is_better,
        )
        # sklearn's search objects take a group-aware splitter *and* the
        # groups themselves; passing only the splitter leaves it calling
        # split() with groups=None, which raises rather than silently
        # ungrouping — but the message names sklearn internals, not the flag.
        if groups is None:
            search.fit(train_x, train_y)
        else:
            search.fit(train_x, train_y, groups=groups[train_index])
        evaluated = max(evaluated, _evaluated(search))
        inner.append(float(search.best_score_))
        winners.append(dict(search.best_params_))
        outer.append(float(scorer(test_y, search.best_estimator_.predict(test_x))))

        # The same model at its defaults, on the same fold — otherwise
        # "tuning gained 0.02" has nothing to be 0.02 against.
        plain = _fold_pipeline(
            features=train_x,
            estimator=build_model(model, task, seed),
            policy=resolved_policy,
            recipe=recipe,
            needs_scaling=entry.needs_scaling,
        )
        plain.fit(train_x, train_y)
        baseline.append(float(scorer(test_y, plain.predict(test_x))))

    best = _most_common(winners)
    return TuneResult(
        model=model,
        strategy=strategy,
        best_params={key.removeprefix("model__"): value for key, value in best.items()},
        candidates=evaluated,
        inner_score=float(np.mean(inner)),
        outer_score=float(np.mean(outer)),
        baseline_score=float(np.mean(baseline)),
        metric=metric_name,
        greater_is_better=greater_is_better,
        seconds=time.perf_counter() - started,
        fold_scores=tuple(outer),
        baseline_folds=tuple(baseline),
    )


def _build_search(
    pipeline: Any,
    space: dict[str, list[Any]],
    *,
    strategy: str,
    candidates: int,
    folds: int,
    seed: int,
    task: str,
    labels: Any,
    groups: Any = None,
    metric: str,
    greater_is_better: bool,
) -> Any:
    """Construct the sklearn search object for one strategy."""
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    from michi.bench.runner import _make_splitter

    scoring = _sklearn_scoring(metric, greater_is_better)
    inner_cv, _ = _make_splitter(task, folds, labels, seed, groups)

    if strategy == "grid":
        return GridSearchCV(pipeline, space, cv=inner_cv, scoring=scoring, n_jobs=1)
    if strategy == "bayes":
        return _bayesian_search(
            pipeline,
            space,
            candidates=candidates,
            cv=inner_cv,
            scoring=scoring,
            seed=seed,
        )
    if strategy == "halving":
        # Successive halving is still behind an explicit opt-in import in
        # scikit-learn; importing it here keeps the failure local and the
        # message honest if that ever changes.
        try:
            from sklearn.experimental import enable_halving_search_cv  # noqa: F401
            from sklearn.model_selection import HalvingRandomSearchCV
        except ImportError as err:  # pragma: no cover - very old sklearn
            msg = (
                "halving search needs a newer scikit-learn; "
                "use --strategy random or --strategy grid"
            )
            raise RunError(msg) from err
        return HalvingRandomSearchCV(
            pipeline,
            space,
            cv=inner_cv,
            scoring=scoring,
            random_state=seed,
            n_jobs=1,
        )
    return RandomizedSearchCV(
        pipeline,
        space,
        n_iter=min(candidates, _grid_size(space)),
        cv=inner_cv,
        scoring=scoring,
        random_state=seed,
        n_jobs=1,
    )


def _sklearn_scoring(metric: str, greater_is_better: bool) -> str:
    """Map michi's metric name onto an sklearn scoring string."""
    mapping = {
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    if metric in mapping:
        return mapping[metric]
    # An unmapped metric is a michi bug, not a user error, and guessing a
    # scorer would silently optimise the wrong thing.
    msg = f"no scikit-learn scorer for metric {metric!r}; michi cannot tune against it"
    raise RunError(msg)


def _grid_size(space: dict[str, list[Any]]) -> int:
    """Total configurations in a space, so a random search never over-draws."""
    total = 1
    for values in space.values():
        total *= max(len(values), 1)
    return total


def _most_common(winners: list[dict[str, Any]]) -> dict[str, Any]:
    """The configuration chosen by the most folds.

    Folds disagree, and picking the winner of one fold would be picking a
    winner by lottery. The configuration most folds agreed on is the one with
    evidence behind it; ties fall to the first, which is stable under a seed.
    """
    if not winners:
        return {}
    counts: dict[str, int] = {}
    for winner in winners:
        key = repr(sorted(winner.items(), key=lambda item: item[0]))
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts, key=lambda key: counts[key])
    for winner in winners:
        if repr(sorted(winner.items(), key=lambda item: item[0])) == best_key:
            return winner
    return winners[0]


def _bayesian_search(
    pipeline: Any,
    space: dict[str, list[Any]],
    *,
    candidates: int,
    cv: Any,
    scoring: str,
    seed: int,
) -> Any:
    """A model-based search over the user's own space.

    Optuna proposes each configuration from what it has learned about the
    objective so far, rather than drawing blind. It may sample only from the
    space it was given — never widen it, never continue past its bounds — so
    what changes is the order candidates are tried in, not which candidates
    exist. ADR-0004 records why that distinction is the one that matters.
    """
    try:
        import optuna
        from optuna.integration import OptunaSearchCV
    except ImportError as err:
        # Never fall back to a different strategy: a user who asked for
        # `bayes` and silently got `random` would be comparing two runs that
        # were never the same experiment.
        msg = (
            "bayesian search needs Optuna, which michi does not install by "
            f"default. {install_hint('bayes')} — or use --strategy random, "
            "halving, or grid, which need nothing extra."
        )
        raise RunError(msg) from err

    # Optuna logs a line per trial at INFO; a tuning run would bury michi's
    # own output under a few hundred of them.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    distributions = {
        name: optuna.distributions.CategoricalDistribution(
            [_hashable(value) for value in values]
        )
        for name, values in space.items()
    }
    return OptunaSearchCV(
        pipeline,
        distributions,
        cv=cv,
        scoring=scoring,
        n_trials=min(candidates, _grid_size(space)),
        random_state=seed,
        verbose=0,
    )


def _hashable(value: Any) -> Any:
    """Optuna's categorical distribution requires hashable choices.

    michi's spaces legitimately contain tuples — a network's layer sizes — and
    those are already hashable; a list would not be, so it is converted rather
    than rejected.
    """
    return tuple(value) if isinstance(value, list) else value


def _evaluated(search: Any) -> int:
    """How many configurations a search actually tried.

    scikit-learn records one entry per candidate under ``cv_results_["params"]``;
    Optuna's wrapper has no such key and exposes ``trials_`` instead. Reading
    only the sklearn shape raised ``KeyError`` the first time a Bayesian search
    finished a fold.
    """
    trials = getattr(search, "trials_", None)
    if trials is not None:
        return len(trials)
    results = getattr(search, "cv_results_", {}) or {}
    params = results.get("params")
    if params is not None:
        return len(params)
    scores = results.get("mean_test_score")
    return len(scores) if scores is not None else 0
