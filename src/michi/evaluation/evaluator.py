"""Evaluating a trained model against a dataset.

Design Principles
-----------------
- **Evaluate models michi never saw trained.** The model is a black box with a
  ``predict`` method; nothing about how it was built is assumed.
- **Always answer "compared to what?".** Trivial baselines run on the same
  rows, every time, and a model that fails to beat them is reported as such
  in plain language.
- **Report where a model fails, not only how often.** Per-slice metrics and
  calibration turn a single score into something actionable.
- **Suspiciously good is a finding.** A perfect score usually means leakage,
  not excellence, and michi says so rather than congratulating the user.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from michi.adapters.model import LoadedModel
from michi.core.artifacts import Finding, Severity
from michi.core.errors import DataError, RunError
from michi.core.io import LoadedTable
from michi.core.manifest import Metric, RunManifest, capture_environment
from michi.evaluation.checks import evaluation_checks
from michi.evaluation.metrics import (
    BOOTSTRAP_SAMPLES,
    classification_metrics,
    detect_task,
    regression_metrics,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

__all__ = ["SliceMetric", "evaluate_model", "new_run_id"]

_MAX_SLICE_COLUMNS: Final = 6
_MAX_SLICE_GROUPS: Final = 10
_MIN_SLICE_ROWS: Final = 20
_CALIBRATION_BINS: Final = 10


@dataclass(frozen=True, slots=True)
class SliceMetric:
    """The headline metric restricted to one subgroup of the data."""

    column: str
    value: str
    n_rows: int
    metric: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "column": self.column,
            "value": self.value,
            "n_rows": self.n_rows,
            "metric": self.metric,
            "score": self.score,
        }


def new_run_id(seed: int = 0) -> str:
    """Return a sortable, unique identifier for a run.

    Examples
    --------
    >>> run_id = new_run_id()
    >>> len(run_id.split("-")) == 2
    True
    """
    import secrets

    from michi.core.artifacts import utc_now_iso

    stamp = utc_now_iso().replace("-", "").replace(":", "")
    return f"{stamp}-{secrets.token_hex(4)}"


def evaluate_model(
    model: LoadedModel,
    table: LoadedTable,
    *,
    target: str,
    task: str | None = None,
    features: tuple[str, ...] | None = None,
    slice_columns: tuple[str, ...] | None = None,
    bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> RunManifest:
    """Evaluate a model and record the result as a run manifest.

    Parameters
    ----------
    model
        The model to evaluate, already loaded.
    table
        Evaluation data, including its provenance.
    target
        Name of the label column.
    task
        ``"classification"`` or ``"regression"``; inferred from the target
        when omitted.
    features
        Columns to pass to the model; defaults to everything but the target.
    slice_columns
        Columns to compute per-group metrics over; low-cardinality columns are
        chosen automatically when omitted.
    bootstrap
        Resamples used for confidence intervals; ``0`` disables them.
    seed
        Seed for bootstrap resampling, recorded in the manifest.

    Returns
    -------
    RunManifest
        The complete, durable record of the evaluation.

    Raises
    ------
    DataError
        If the target is missing from the data, or holds no usable values.
    RunError
        If the model's predictions cannot be aligned with the labels.
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

    feature_frame = _select_features(usable, target, features)
    truth = np.asarray(usable[target])
    resolved_task = task or detect_task(truth)
    if resolved_task not in {"classification", "regression"}:
        msg = f"unknown task {resolved_task!r}; expected classification or regression"
        raise RunError(msg)

    predictions = model.predict(feature_frame)
    if predictions.shape[0] != truth.shape[0]:
        msg = (
            f"model returned {predictions.shape[0]} predictions for "
            f"{truth.shape[0]} rows"
        )
        raise RunError(msg)
    predictions = np.asarray(predictions).reshape(truth.shape[0], -1)[:, 0]

    probabilities = (
        model.predict_proba(feature_frame)
        if resolved_task == "classification"
        else None
    )

    # sklearn warns about degenerate conditions — a single class present, a
    # label the model never predicts — that michi reports itself, as checks,
    # in language the user can act on. Its warnings would only duplicate them
    # in a form that points at library internals.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _measure(
            model=model,
            table=table,
            usable=usable,
            feature_frame=feature_frame,
            truth=truth,
            predictions=predictions,
            probabilities=probabilities,
            target=target,
            task=resolved_task,
            slice_columns=slice_columns,
            bootstrap=bootstrap,
            seed=seed,
            started=started,
        )


def _measure(
    *,
    model: LoadedModel,
    table: LoadedTable,
    usable: pd.DataFrame,
    feature_frame: pd.DataFrame,
    truth: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any] | None,
    target: str,
    task: str,
    slice_columns: tuple[str, ...] | None,
    bootstrap: int,
    seed: int,
    started: float,
) -> RunManifest:
    """Compute every metric, baseline, slice, and check for one evaluation."""
    resolved_task = task

    if resolved_task == "classification":
        truth, predictions = _align_labels(truth, predictions)
        metrics = classification_metrics(
            truth, predictions, probabilities, bootstrap=bootstrap, seed=seed
        )
    else:
        truth = truth.astype(float)
        predictions = predictions.astype(float)
        metrics = regression_metrics(truth, predictions, bootstrap=bootstrap, seed=seed)

    baselines = _baseline_metrics(truth, resolved_task, seed)
    slices = _slice_metrics(
        usable, truth, predictions, resolved_task, target, slice_columns
    )
    details: dict[str, Any] = {
        "slices": [item.to_dict() for item in slices],
        "n_features": int(feature_frame.shape[1]),
        "features": [str(name) for name in feature_frame.columns],
    }

    if resolved_task == "classification":
        classes, matrix = _confusion(truth, predictions)
        details["classes"] = classes
        details["confusion"] = matrix
        details["class_counts"] = _class_counts(truth)
        calibration = _calibration(truth, probabilities, model)
        if calibration is not None:
            details["calibration"] = calibration["bins"]
            details["ece"] = calibration["ece"]

    checks = evaluation_checks(
        task=resolved_task,
        metrics=metrics,
        baselines=baselines,
        slices=slices,
        details=details,
        n_rows=int(truth.shape[0]),
    )

    return RunManifest(
        run_id=new_run_id(seed),
        kind="eval",
        dataset=table.source,
        target=target,
        task=resolved_task,
        model=model.spec,
        metrics=metrics,
        baselines=baselines,
        checks=checks,
        details=details,
        seed=seed,
        n_rows=int(truth.shape[0]),
        duration_s=time.perf_counter() - started,
        environment=capture_environment(),
    )


def _select_features(
    frame: pd.DataFrame, target: str, features: tuple[str, ...] | None
) -> pd.DataFrame:
    """Choose the columns handed to the model."""
    if features is None:
        return frame.drop(columns=[target])
    missing = [name for name in features if name not in frame.columns]
    if missing:
        msg = f"requested features not in the data: {', '.join(missing)}"
        raise DataError(msg)
    return frame[list(features)]


def _align_labels(
    truth: np.ndarray[Any, Any], predictions: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Put labels and predictions in a comparable representation.

    A model may return strings where the labels are integers, or vice versa.
    Comparing them as text is the only representation that always works and
    never silently mismatches ``1`` with ``"1"``.
    """
    import numpy as np

    if truth.dtype == predictions.dtype:
        return truth, predictions
    return np.asarray([str(item) for item in truth]), np.asarray(
        [str(item) for item in predictions]
    )


def _baseline_metrics(
    truth: np.ndarray[Any, Any], task: str, seed: int
) -> dict[str, tuple[Metric, ...]]:
    """Score trivial models on the same labels.

    These are the answer to "compared to what?". A model that cannot beat
    them has learned nothing, however respectable its raw score looks.
    """
    import numpy as np

    baselines: dict[str, tuple[Metric, ...]] = {}
    if task == "classification":
        values, counts = np.unique(truth, return_counts=True)
        majority = values[int(np.argmax(counts))]
        constant = np.full(truth.shape, majority)
        baselines["most_frequent"] = classification_metrics(
            truth, constant, None, bootstrap=0, seed=seed
        )
        rng = np.random.default_rng(seed)
        baselines["stratified"] = classification_metrics(
            truth,
            rng.choice(values, size=truth.shape[0], p=counts / counts.sum()),
            None,
            bootstrap=0,
            seed=seed,
        )
    else:
        baselines["mean"] = regression_metrics(
            truth, np.full(truth.shape, float(np.mean(truth))), bootstrap=0, seed=seed
        )
        baselines["median"] = regression_metrics(
            truth,
            np.full(truth.shape, float(np.median(truth))),
            bootstrap=0,
            seed=seed,
        )
    return baselines


def _slice_metrics(
    frame: pd.DataFrame,
    truth: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    task: str,
    target: str,
    slice_columns: tuple[str, ...] | None,
) -> tuple[SliceMetric, ...]:
    """Score the model separately within each subgroup of chosen columns.

    An aggregate metric hides which subgroup a model fails on. Slices are the
    cheapest way to surface that, so michi computes them by default over
    low-cardinality columns.
    """
    import numpy as np
    from sklearn import metrics as skm

    columns = (
        list(slice_columns)
        if slice_columns is not None
        else _auto_slice_columns(frame, target)
    )
    if not columns:
        return ()

    if task == "classification":
        metric_name = "balanced_accuracy"

        def score(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
            return float(skm.balanced_accuracy_score(y_true, y_pred))
    else:
        metric_name = "rmse"

        def score(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
            return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))

    results: list[SliceMetric] = []
    for column in columns:
        if column not in frame.columns:
            msg = f"slice column {column!r} is not in the data"
            raise DataError(msg)
        series = frame[column]
        for value in series.dropna().unique()[:_MAX_SLICE_GROUPS]:
            mask = np.asarray(series == value)
            if int(mask.sum()) < _MIN_SLICE_ROWS:
                continue
            try:
                results.append(
                    SliceMetric(
                        column=str(column),
                        value=str(value),
                        n_rows=int(mask.sum()),
                        metric=metric_name,
                        score=score(truth[mask], predictions[mask]),
                    )
                )
            except (ValueError, IndexError):
                continue
    return tuple(results)


def _auto_slice_columns(frame: pd.DataFrame, target: str) -> list[str]:
    """Pick low-cardinality columns worth slicing on."""
    chosen: list[str] = []
    for name in frame.columns:
        if str(name) == target:
            continue
        try:
            unique = int(frame[name].nunique(dropna=True))
        except TypeError:
            continue
        if 2 <= unique <= _MAX_SLICE_GROUPS:
            chosen.append(str(name))
        if len(chosen) >= _MAX_SLICE_COLUMNS:
            break
    return chosen


def _confusion(
    truth: np.ndarray[Any, Any], predictions: np.ndarray[Any, Any]
) -> tuple[list[str], list[list[int]]]:
    """Return class labels and the confusion matrix as plain JSON data."""
    import numpy as np
    from sklearn import metrics as skm

    classes = np.unique(np.concatenate([truth, predictions]))
    matrix = skm.confusion_matrix(truth, predictions, labels=classes)
    return [str(item) for item in classes], [
        [int(cell) for cell in row] for row in matrix
    ]


def _class_counts(truth: np.ndarray[Any, Any]) -> dict[str, int]:
    """Count how many rows carry each class."""
    import numpy as np

    values, counts = np.unique(truth, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


def _calibration(
    truth: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any] | None,
    model: LoadedModel,
) -> dict[str, Any] | None:
    """Bin predicted probabilities against observed frequencies.

    A well-calibrated model that says "70% likely" is right about 70% of the
    time. Calibration is invisible to accuracy but decides whether a
    probability can be used as one.
    """
    import numpy as np

    if probabilities is None or probabilities.ndim != 2 or probabilities.shape[1] != 2:
        return None

    classes = model.classes
    positive = classes[-1] if classes is not None else np.unique(truth)[-1]
    observed = (truth == positive).astype(float)
    if observed.sum() == 0 or observed.sum() == observed.shape[0]:
        return None

    scores = probabilities[:, 1]
    edges = np.linspace(0.0, 1.0, _CALIBRATION_BINS + 1)
    bins: list[list[float]] = []
    error = 0.0
    for index in range(_CALIBRATION_BINS):
        low, high = edges[index], edges[index + 1]
        mask = (scores >= low) & (
            scores <= high if index == _CALIBRATION_BINS - 1 else scores < high
        )
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(scores[mask].mean())
        observed_rate = float(observed[mask].mean())
        bins.append([mean_predicted, observed_rate, count])
        error += (count / scores.shape[0]) * abs(mean_predicted - observed_rate)

    if not bins:
        return None
    return {"bins": bins, "ece": round(error, 4)}


def summarise_checks(checks: tuple[Finding, ...]) -> str:
    """One-line summary of the most severe check, for terse output."""
    if not checks:
        return "no checks raised"
    worst = min(checks, key=lambda check: check.severity.rank)
    prefix = "" if worst.severity is Severity.INFO else f"{worst.severity.value}: "
    return f"{prefix}{worst.summary}"
