"""Computing metrics, with uncertainty and baselines attached.

Design Principles
-----------------
- **A number without an interval invites false confidence.** Every headline
  metric is reported with a bootstrap confidence interval, because the
  difference between 0.91 and 0.89 is meaningless when the interval spans
  0.06.
- **Baselines are not optional.** michi always evaluates trivial models on the
  same data, so "is this good?" has an answer that does not depend on
  intuition about the metric's scale.
- **The task decides the metrics.** Classification and regression have
  separate, complete metric sets; michi never reports accuracy for a
  regression or R² for a classifier.
- sklearn is an implementation detail: this module returns michi
  :class:`~michi.core.manifest.Metric` objects and never leaks estimator
  types.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Final

from michi.core.manifest import Metric

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "BOOTSTRAP_SAMPLES",
    "classification_metrics",
    "detect_task",
    "regression_metrics",
]

BOOTSTRAP_SAMPLES: Final = 1_000
"""Default resamples used to estimate confidence intervals."""

_MAX_CLASSES_FOR_CLASSIFICATION: Final = 50
_CONTINUOUS_UNIQUE_RATIO: Final = 0.05


def detect_task(target: np.ndarray[Any, Any]) -> str:
    """Infer whether a target describes classification or regression.

    Non-numeric targets are always classification. A numeric target is treated
    as classification when it holds few distinct values relative to its
    length — the shape of an encoded label rather than a measured quantity.

    Examples
    --------
    >>> import numpy as np
    >>> detect_task(np.array([0, 1, 1, 0, 1]))
    'classification'
    >>> detect_task(np.arange(500) * 1.5)
    'regression'
    """
    import numpy as np

    values = np.asarray(target)
    if values.dtype.kind in {"U", "S", "O", "b"}:
        return "classification"

    finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if finite.size == 0:
        return "classification"

    unique = np.unique(finite)
    if unique.size > _MAX_CLASSES_FOR_CLASSIFICATION:
        return "regression"
    if values.dtype.kind == "f" and not np.all(finite == np.round(finite)):
        return "regression"
    if unique.size <= 2:
        return "classification"
    return (
        "regression"
        if unique.size / finite.size > _CONTINUOUS_UNIQUE_RATIO
        else "classification"
    )


def classification_metrics(
    truth: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any] | None = None,
    *,
    bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[Metric, ...]:
    """Compute classification metrics with bootstrap intervals.

    The first metric returned is the headline one: balanced accuracy, chosen
    over plain accuracy because it does not flatter a model that has merely
    learned the majority class.
    """
    import numpy as np
    from sklearn import metrics as skm

    classes = np.unique(np.concatenate([truth, predictions]))
    binary = classes.size <= 2
    average = "binary" if binary else "macro"
    positive = classes[-1] if binary else None

    def _f1(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        return float(
            skm.f1_score(
                y_true,
                y_pred,
                average=average,
                pos_label=positive,
                zero_division=0,
            )
        )

    def _precision(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        return float(
            skm.precision_score(
                y_true,
                y_pred,
                average=average,
                pos_label=positive,
                zero_division=0,
            )
        )

    def _recall(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        return float(
            skm.recall_score(
                y_true,
                y_pred,
                average=average,
                pos_label=positive,
                zero_division=0,
            )
        )

    scorers: list[tuple[str, Any]] = [
        ("balanced_accuracy", skm.balanced_accuracy_score),
        ("accuracy", skm.accuracy_score),
        ("f1", _f1),
        ("precision", _precision),
        ("recall", _recall),
        ("mcc", skm.matthews_corrcoef),
    ]

    results: list[Metric] = []
    for name, scorer in scorers:
        results.append(_bootstrapped(name, truth, predictions, scorer, bootstrap, seed))

    results.extend(
        _probability_metrics(truth, probabilities, classes, binary, bootstrap, seed)
    )
    return tuple(results)


def regression_metrics(
    truth: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    *,
    bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[Metric, ...]:
    """Compute regression metrics with bootstrap intervals.

    The headline metric is RMSE: expressed in the units of the target, which
    makes "is this error acceptable?" a question a domain expert can answer.
    """
    import numpy as np
    from sklearn import metrics as skm

    def _rmse(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))

    def _mape(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        mask = y_true != 0
        if not mask.any():
            return float("nan")
        return float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
        )

    scorers: list[tuple[str, Any, bool]] = [
        ("rmse", _rmse, False),
        ("mae", skm.mean_absolute_error, False),
        ("median_ae", skm.median_absolute_error, False),
        ("r2", skm.r2_score, True),
        ("mape", _mape, False),
    ]

    return tuple(
        _bootstrapped(
            name,
            truth,
            predictions,
            scorer,
            bootstrap,
            seed,
            greater_is_better=higher_better,
        )
        for name, scorer, higher_better in scorers
    )


def _probability_metrics(
    truth: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any] | None,
    classes: np.ndarray[Any, Any],
    binary: bool,
    bootstrap: int,
    seed: int,
) -> list[Metric]:
    """Metrics that need predicted probabilities, when the model offers them."""
    import numpy as np
    from sklearn import metrics as skm

    if probabilities is None or probabilities.ndim != 2:
        return []

    results: list[Metric] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if binary and probabilities.shape[1] == 2:
                scores = probabilities[:, 1]
                positive = classes[-1]
                binary_truth = (truth == positive).astype(int)
                results.append(
                    _bootstrapped(
                        "roc_auc",
                        binary_truth,
                        scores,
                        skm.roc_auc_score,
                        bootstrap,
                        seed,
                    )
                )
                results.append(
                    _bootstrapped(
                        "pr_auc",
                        binary_truth,
                        scores,
                        skm.average_precision_score,
                        bootstrap,
                        seed,
                    )
                )
                results.append(
                    Metric(
                        "brier",
                        float(skm.brier_score_loss(binary_truth, scores)),
                        greater_is_better=False,
                    )
                )
            elif not binary and probabilities.shape[1] == classes.size:
                results.append(
                    Metric(
                        "roc_auc_ovr",
                        float(
                            skm.roc_auc_score(
                                truth,
                                probabilities,
                                multi_class="ovr",
                                average="macro",
                                labels=list(classes),
                            )
                        ),
                    )
                )
            results.append(
                Metric(
                    "log_loss",
                    float(skm.log_loss(truth, probabilities, labels=list(classes))),
                    greater_is_better=False,
                )
            )
        except (ValueError, IndexError):
            # A degenerate split — one class present, or a probability matrix
            # that does not line up with the labels — costs the probability
            # metrics, never the run.
            return results
    return [metric for metric in results if np.isfinite(metric.value)]


def _bootstrapped(
    name: str,
    truth: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    scorer: Any,
    resamples: int,
    seed: int,
    *,
    greater_is_better: bool = True,
) -> Metric:
    """Score once, then resample to estimate a percentile interval.

    Resampling rows with replacement estimates how much the score depends on
    which rows happened to be in the test set. It cannot capture uncertainty
    from the training sample — that needs cross-validation, which arrives with
    ``michi bench``.
    """
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = float(scorer(truth, predictions))
        except (ValueError, IndexError):
            return Metric(name, float("nan"), greater_is_better=greater_is_better)

        if resamples <= 0 or truth.shape[0] < 20:
            return Metric(name, value, greater_is_better=greater_is_better)

        rng = np.random.default_rng(seed)
        size = truth.shape[0]
        samples: list[float] = []
        for _ in range(resamples):
            index = rng.integers(0, size, size)
            try:
                samples.append(float(scorer(truth[index], predictions[index])))
            except (ValueError, IndexError):
                continue

    finite = [item for item in samples if item == item]
    if len(finite) < resamples // 2:
        return Metric(name, value, greater_is_better=greater_is_better)

    low, high = np.percentile(finite, [2.5, 97.5])
    return Metric(
        name,
        value,
        ci_low=float(low),
        ci_high=float(high),
        greater_is_better=greater_is_better,
    )
