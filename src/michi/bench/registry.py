"""The catalogue of models ``michi bench`` can train.

Design Principles
-----------------
- **A menu, not a recommendation.** michi lists what is available with a
  one-line description of each; which to try is the user's choice. Nothing is
  selected automatically and no model is called "best".
- **Nothing downloads.** Every entry is an algorithm that trains locally on
  the user's data in seconds. michi makes no network calls, ever.
- **Heavy libraries stay optional.** Gradient-boosting packages live behind
  the ``bench`` extra and are imported only when actually requested, with an
  error naming the exact install command when they are missing.
- Entries are data, not classes: adding a model is one table entry, not a
  subclass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final

from michi.core.errors import RunError, install_hint

__all__ = [
    "ModelEntry",
    "available_models",
    "build_model",
    "model_entry",
    "register_transient",
]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One entry in the model catalogue.

    Attributes
    ----------
    name
        Short identifier used on the command line.
    tasks
        Which tasks the model supports.
    summary
        One factual line about the model's behaviour — never a judgement of
        quality.
    extra
        The michi extra that must be installed, if any.
    needs_scaling
        Whether the model is sensitive to feature scale, which decides
        whether the default preprocessing standardises numeric columns.
    """

    name: str
    tasks: frozenset[str]
    summary: str
    factory: Callable[[str, int], Any]
    extra: str | None = None
    needs_scaling: bool = False

    def supports(self, task: str) -> bool:
        """Whether this entry can be trained for the given task."""
        return task in self.tasks


_BOTH: Final = frozenset({"classification", "regression"})
_CLASSIFICATION: Final = frozenset({"classification"})
_REGRESSION: Final = frozenset({"regression"})


def _linear(task: str, seed: int) -> Any:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task == "classification":
        return LogisticRegression(max_iter=1000, random_state=seed)
    return LinearRegression()


def _ridge(task: str, seed: int) -> Any:
    from sklearn.linear_model import Ridge, RidgeClassifier

    return (
        RidgeClassifier(random_state=seed)
        if task == "classification"
        else Ridge(random_state=seed)
    )


def _lasso(task: str, seed: int) -> Any:
    from sklearn.linear_model import Lasso

    return Lasso(random_state=seed)


def _random_forest(task: str, seed: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    kwargs = {"n_estimators": 300, "random_state": seed, "n_jobs": -1}
    return (
        RandomForestClassifier(**kwargs)
        if task == "classification"
        else RandomForestRegressor(**kwargs)
    )


def _extra_trees(task: str, seed: int) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    kwargs = {"n_estimators": 300, "random_state": seed, "n_jobs": -1}
    return (
        ExtraTreesClassifier(**kwargs)
        if task == "classification"
        else ExtraTreesRegressor(**kwargs)
    )


def _hist_gbm(task: str, seed: int) -> Any:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    return (
        HistGradientBoostingClassifier(random_state=seed)
        if task == "classification"
        else HistGradientBoostingRegressor(random_state=seed)
    )


def _decision_tree(task: str, seed: int) -> Any:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    return (
        DecisionTreeClassifier(random_state=seed)
        if task == "classification"
        else DecisionTreeRegressor(random_state=seed)
    )


def _knn(task: str, seed: int) -> Any:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    if task == "classification":
        return KNeighborsClassifier()
    return KNeighborsRegressor()


def _svm(task: str, seed: int) -> Any:
    from sklearn.svm import SVC, SVR

    if task == "classification":
        return SVC(probability=True, random_state=seed)
    return SVR()


def _naive_bayes(task: str, seed: int) -> Any:
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB()


def _dummy(task: str, seed: int) -> Any:
    from sklearn.dummy import DummyClassifier, DummyRegressor

    return (
        DummyClassifier(strategy="most_frequent")
        if task == "classification"
        else DummyRegressor(strategy="mean")
    )


def _xgboost(task: str, seed: int) -> Any:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as err:
        raise _missing_extra("xgboost", "bench") from err

    kwargs = {"random_state": seed, "n_jobs": -1, "verbosity": 0}
    if task == "classification":
        return XGBClassifier(**kwargs)
    return XGBRegressor(**kwargs)


def _lightgbm(task: str, seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as err:
        raise _missing_extra("lightgbm", "bench") from err

    kwargs = {"random_state": seed, "n_jobs": -1, "verbose": -1}
    if task == "classification":
        return LGBMClassifier(**kwargs)
    return LGBMRegressor(**kwargs)


def _catboost(task: str, seed: int) -> Any:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as err:
        raise _missing_extra("catboost", "bench") from err

    kwargs = {"random_seed": seed, "verbose": False, "allow_writing_files": False}
    return (
        CatBoostClassifier(**kwargs)
        if task == "classification"
        else CatBoostRegressor(**kwargs)
    )


def _mlp(task: str, seed: int) -> Any:
    from michi.bench.neural import build_mlp

    return build_mlp(task, seed)


def _torch_mlp(task: str, seed: int) -> Any:
    from michi.bench.neural import build_torch_mlp

    return build_torch_mlp(task, seed)


def _missing_extra(package: str, extra: str) -> RunError:
    return RunError(
        f"{package} is not installed. Install it with: {install_hint(extra)}"
    )


_REGISTRY: Final[tuple[ModelEntry, ...]] = (
    ModelEntry(
        name="dummy",
        tasks=_BOTH,
        summary=(
            "predicts the most frequent class or the mean; "
            "the floor any model must clear"
        ),
        factory=_dummy,
    ),
    ModelEntry(
        name="linear",
        tasks=_BOTH,
        summary=(
            "logistic or ordinary least squares regression; fast, and its "
            "coefficients read directly"
        ),
        factory=_linear,
        needs_scaling=True,
    ),
    ModelEntry(
        name="ridge",
        tasks=_BOTH,
        summary=(
            "linear model with L2 penalty; steadier than plain linear when "
            "features are correlated"
        ),
        factory=_ridge,
        needs_scaling=True,
    ),
    ModelEntry(
        name="lasso",
        tasks=_REGRESSION,
        summary=(
            "linear regression with L1 penalty; drives some coefficients to "
            "exactly zero"
        ),
        factory=_lasso,
        needs_scaling=True,
    ),
    ModelEntry(
        name="tree",
        tasks=_BOTH,
        summary=(
            "a single decision tree; readable end to end, and prone to "
            "overfitting alone"
        ),
        factory=_decision_tree,
    ),
    ModelEntry(
        name="rf",
        tasks=_BOTH,
        summary=("random forest of 300 trees; a common starting point on tabular data"),
        factory=_random_forest,
    ),
    ModelEntry(
        name="extra-trees",
        tasks=_BOTH,
        summary=(
            "randomised forest variant; splits at random thresholds, trains "
            "faster than rf"
        ),
        factory=_extra_trees,
    ),
    ModelEntry(
        name="hist-gbm",
        tasks=_BOTH,
        summary=(
            "histogram gradient boosting from sklearn; handles missing values "
            "natively, no extra needed"
        ),
        factory=_hist_gbm,
    ),
    ModelEntry(
        name="knn",
        tasks=_BOTH,
        summary=(
            "k-nearest neighbours; no training step, prediction cost grows "
            "with the data"
        ),
        factory=_knn,
        needs_scaling=True,
    ),
    ModelEntry(
        name="svm",
        tasks=_BOTH,
        summary=(
            "support vector machine with an RBF kernel; scales poorly beyond "
            "tens of thousands of rows"
        ),
        factory=_svm,
        needs_scaling=True,
    ),
    ModelEntry(
        name="naive-bayes",
        tasks=_CLASSIFICATION,
        summary=(
            "Gaussian naive Bayes; assumes independent features, trains "
            "almost instantly"
        ),
        factory=_naive_bayes,
        needs_scaling=True,
    ),
    ModelEntry(
        name="xgb",
        tasks=_BOTH,
        summary="XGBoost gradient boosting",
        factory=_xgboost,
        extra="bench",
    ),
    ModelEntry(
        name="lgbm",
        tasks=_BOTH,
        summary=("LightGBM gradient boosting; leaf-wise growth, fast on wide data"),
        factory=_lightgbm,
        extra="bench",
    ),
    ModelEntry(
        name="catboost",
        tasks=_BOTH,
        summary=(
            "CatBoost gradient boosting; ordered boosting, strong categorical handling"
        ),
        factory=_catboost,
        extra="bench",
    ),
    ModelEntry(
        name="mlp",
        tasks=_BOTH,
        summary=(
            "feed-forward neural network, two hidden layers; trains with "
            "early stopping and needs no extra install"
        ),
        factory=_mlp,
        needs_scaling=True,
    ),
    ModelEntry(
        name="torch-mlp",
        tasks=_BOTH,
        summary=(
            "PyTorch network; the epochs, Adam, batching, and early stopping "
            "written once so you stop retyping them"
        ),
        factory=_torch_mlp,
        needs_scaling=True,
        extra="torch",
    ),
)


_TRANSIENT: dict[str, ModelEntry] = {}
"""Entries registered for the lifetime of one command.

An ensemble is assembled from models the user named, so it cannot be a
constant in the catalogue. Registering it briefly lets `run_benchmark` treat
it as an ordinary model — same folds, same baseline, same significance test —
without a single branch anywhere downstream asking "is this an ensemble?".
"""


@contextmanager
def register_transient(
    name: str,
    *,
    summary: str,
    factory: Any,
    needs_scaling: bool = False,
    tasks: frozenset[str] = _BOTH,
) -> Iterator[None]:
    """Add a catalogue entry for the duration of a `with` block.

    Raises
    ------
    RunError
        If the name would shadow a permanent catalogue entry, which would make
        `--models rf` mean different things in different commands.
    """
    if any(entry.name == name for entry in _REGISTRY):
        msg = f"{name!r} is already a catalogue model and cannot be redefined"
        raise RunError(msg)

    _TRANSIENT[name] = ModelEntry(
        name=name,
        tasks=tasks,
        summary=summary,
        factory=factory,
        needs_scaling=needs_scaling,
    )
    try:
        yield
    finally:
        _TRANSIENT.pop(name, None)


def _plugin_models() -> tuple[ModelEntry, ...]:
    """Models contributed by installed plugins.

    A plugin may not shadow a built-in: a user reading ``--list-models`` has
    to be able to trust that ``rf`` means what the documentation says.
    """
    from michi.plugins.registry import MODEL_GROUP, discover

    builtin = {entry.name for entry in _REGISTRY}
    extra: list[ModelEntry] = []
    for _, loaded, record in discover(MODEL_GROUP):
        if not record.loaded or loaded is None:
            continue
        candidates = loaded() if callable(loaded) else loaded
        if isinstance(candidates, ModelEntry):
            candidates = (candidates,)
        try:
            for entry in candidates:
                if isinstance(entry, ModelEntry) and entry.name not in builtin:
                    extra.append(entry)
                    builtin.add(entry.name)
        except TypeError:
            continue
    return tuple(extra)


def available_models(task: str | None = None) -> tuple[ModelEntry, ...]:
    """List the catalogue, optionally narrowed to one task.

    Includes models contributed by installed plugins, which are appended
    after the built-ins and can never shadow one.

    Examples
    --------
    >>> names = [entry.name for entry in available_models("regression")]
    >>> "lasso" in names and "naive-bayes" not in names
    True
    """
    entries = _REGISTRY + _plugin_models() + tuple(_TRANSIENT.values())
    if task is None:
        return entries
    return tuple(entry for entry in entries if entry.supports(task))


def model_entry(name: str) -> ModelEntry:
    """Look up one catalogue entry by name.

    Raises
    ------
    RunError
        If no such model exists, listing what does.
    """
    for entry in available_models():
        if entry.name == name:
            return entry
    known = ", ".join(item.name for item in available_models())
    msg = f"unknown model {name!r}. Available models: {known}"
    raise RunError(msg)


def build_model(name: str, task: str, seed: int) -> Any:
    """Instantiate a catalogue model for a task.

    Raises
    ------
    RunError
        If the model does not support the task, or its extra is missing.
    """
    entry = model_entry(name)
    if not entry.supports(task):
        supported = ", ".join(sorted(entry.tasks))
        msg = f"{name!r} does not support {task}; it supports: {supported}"
        raise RunError(msg)
    return entry.factory(task, seed)
