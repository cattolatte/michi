"""Computing a dataset profile.

Design Principles
-----------------
- **Describe, never prescribe.** The profiler reports what is measurably true
  about the data. Deciding what to do about it is the user's job, and the
  options are offered separately by the explanation layer.
- **michi's own taxonomy.** Columns are classified as numeric, categorical,
  boolean, datetime, text, or empty — the categories users think in — rather
  than surfacing raw storage dtypes as the primary signal.
- **Every statistic is JSON-safe.** Profiles must round-trip through a file
  without depending on the dataframe library that produced them, so datetimes
  are stored as epoch seconds and rendered by the report layer.
- **Bounded cost.** Expensive pairwise analysis (correlation, duplicate
  detection) is capped so that profiling a very wide dataset stays usable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from michi.core.artifacts import ColumnKind, ColumnProfile, DatasetProfile
from michi.core.errors import DataError
from michi.core.io import LoadedTable
from michi.inspection.findings import Thresholds, detect_findings

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["profile_table"]

_TOP_VALUES: Final = 5
_TEXT_MEAN_LENGTH: Final = 50.0
_TEXT_UNIQUE_RATIO: Final = 0.9
_HISTOGRAM_BINS: Final = 20


def profile_table(
    table: LoadedTable,
    *,
    target: str | None = None,
    thresholds: Thresholds | None = None,
) -> DatasetProfile:
    """Profile a loaded table and attach the findings it warrants.

    Parameters
    ----------
    table
        Data plus provenance, as produced by
        :func:`michi.core.io.load_table`.
    target
        Optional name of the label column. When given, michi additionally
        looks for class imbalance and target-leakage suspects.
    thresholds
        Detection thresholds; the documented defaults are used when omitted.

    Returns
    -------
    DatasetProfile
        The complete profile artifact.

    Raises
    ------
    DataError
        If ``target`` is not a column of the dataset.
    """
    frame = table.frame
    if target is not None and target not in frame.columns:
        available = ", ".join(str(name) for name in frame.columns[:10])
        msg = f"target {target!r} is not a column; available columns: {available}"
        raise DataError(msg)

    columns = tuple(_profile_column(frame[name], str(name)) for name in frame.columns)
    duplicate_rows = _count_duplicate_rows(frame)
    findings = detect_findings(
        frame,
        columns=columns,
        target=target,
        duplicate_rows=duplicate_rows,
        thresholds=thresholds or Thresholds(),
    )

    return DatasetProfile(
        source=table.source,
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        duplicate_rows=duplicate_rows,
        columns=columns,
        findings=findings,
        target=target,
    )


def _count_duplicate_rows(frame: pd.DataFrame) -> int:
    """Count fully duplicated rows, tolerating unhashable cell values."""
    try:
        return int(frame.duplicated().sum())
    except TypeError:
        return int(frame.astype(str).duplicated().sum())


def _profile_column(series: pd.Series[Any], name: str) -> ColumnProfile:
    """Compute the profile of a single column."""
    import pandas as pd

    total = int(series.shape[0])
    missing = int(series.isna().sum())
    count = total - missing
    non_null = series.dropna()

    try:
        unique = int(non_null.nunique())
    except TypeError:
        non_null = non_null.astype(str)
        unique = int(non_null.nunique())

    kind = _infer_kind(series, non_null, count=count, unique=unique)
    dtype = str(series.dtype)

    stats: dict[str, float] = {}
    top_values: tuple[tuple[str, int], ...] = ()
    histogram: tuple[tuple[float, float, int], ...] = ()

    if kind is ColumnKind.NUMERIC and count:
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        stats = _numeric_stats(numeric)
        histogram = _histogram(numeric)
    elif kind is ColumnKind.DATETIME and count:
        stats = _datetime_stats(non_null)
    elif (
        kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN, ColumnKind.TEXT} and count
    ):
        top_values = _top_values(non_null)
        if kind is ColumnKind.TEXT:
            stats = _text_stats(non_null)

    return ColumnProfile(
        name=name,
        kind=kind,
        dtype=dtype,
        count=count,
        missing=missing,
        unique=unique,
        stats=stats,
        top_values=top_values,
        histogram=histogram,
    )


def _infer_kind(
    series: pd.Series[Any],
    non_null: pd.Series[Any],
    *,
    count: int,
    unique: int,
) -> ColumnKind:
    """Map a column onto michi's column taxonomy."""
    import pandas as pd

    if count == 0:
        return ColumnKind.EMPTY
    if pd.api.types.is_bool_dtype(series.dtype):
        return ColumnKind.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return ColumnKind.DATETIME
    if pd.api.types.is_numeric_dtype(series.dtype):
        return ColumnKind.NUMERIC
    if isinstance(series.dtype, pd.CategoricalDtype):
        return ColumnKind.CATEGORICAL

    # Object-like: distinguish free text from categories by shape, not dtype.
    sample = non_null.astype(str).head(1000)
    mean_length = float(sample.str.len().mean()) if not sample.empty else 0.0
    unique_ratio = unique / count if count else 0.0
    if mean_length > _TEXT_MEAN_LENGTH or (
        unique_ratio > _TEXT_UNIQUE_RATIO and count > 100
    ):
        return ColumnKind.TEXT
    return ColumnKind.CATEGORICAL


def _numeric_stats(values: pd.Series[Any]) -> dict[str, float]:
    """Descriptive statistics for a numeric column."""
    if values.empty:
        return {}
    quantiles = values.quantile([0.25, 0.5, 0.75])
    q1 = float(quantiles.iloc[0])
    median = float(quantiles.iloc[1])
    q3 = float(quantiles.iloc[2])
    iqr = q3 - q1
    std = values.std()
    skew = values.skew()
    kurtosis = values.kurtosis()

    stats: dict[str, float] = {
        "mean": float(values.mean()),
        "std": 0.0 if _is_missing(std) else float(std),
        "min": float(values.min()),
        "p25": q1,
        "median": median,
        "p75": q3,
        "max": float(values.max()),
        "iqr": iqr,
        "zeros": float((values == 0).sum()),
        "negatives": float((values < 0).sum()),
    }
    skew_value = _as_float(skew)
    if skew_value is not None:
        stats["skew"] = skew_value
    kurtosis_value = _as_float(kurtosis)
    if kurtosis_value is not None:
        stats["kurtosis"] = kurtosis_value
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        stats["outliers"] = float(((values < lower) | (values > upper)).sum())
        stats["outlier_low"] = lower
        stats["outlier_high"] = upper
    else:
        stats["outliers"] = 0.0
    return stats


def _histogram(
    values: pd.Series[Any], bins: int = _HISTOGRAM_BINS
) -> tuple[tuple[float, float, int], ...]:
    """Bin a numeric column so every renderer draws the same distribution."""
    import numpy as np

    if values.empty:
        return ()
    finite = values[np.isfinite(values)]
    if finite.empty:
        return ()
    low = float(finite.min())
    high = float(finite.max())
    if low == high:
        return ((low, high, int(finite.shape[0])),)
    counts, edges = np.histogram(finite.to_numpy(), bins=bins)
    return tuple(
        (float(edges[index]), float(edges[index + 1]), int(count))
        for index, count in enumerate(counts)
    )


def _datetime_stats(values: pd.Series[Any]) -> dict[str, float]:
    """Range statistics for a datetime column, stored as epoch seconds."""
    try:
        minimum = values.min()
        maximum = values.max()
        min_epoch = float(minimum.timestamp())
        max_epoch = float(maximum.timestamp())
    except (AttributeError, ValueError, OSError, OverflowError):
        return {}
    return {
        "min_epoch_s": min_epoch,
        "max_epoch_s": max_epoch,
        "span_days": (max_epoch - min_epoch) / 86400.0,
    }


def _text_stats(values: pd.Series[Any]) -> dict[str, float]:
    """Length statistics for a free-text column."""
    lengths = values.astype(str).str.len()
    if lengths.empty:
        return {}
    return {
        "mean_length": float(lengths.mean()),
        "min_length": float(lengths.min()),
        "max_length": float(lengths.max()),
    }


def _top_values(values: pd.Series[Any]) -> tuple[tuple[str, int], ...]:
    """Return the most frequent values of a column."""
    try:
        counts = values.value_counts().head(_TOP_VALUES)
    except TypeError:
        counts = values.astype(str).value_counts().head(_TOP_VALUES)
    return tuple((str(value), int(count)) for value, count in counts.items())


def _is_missing(value: Any) -> bool:
    """Whether a scalar produced by pandas is NaN or NA."""
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    """Convert a pandas scalar to ``float``, or ``None`` if it is not numeric.

    pandas reductions are typed as broad unions (a statistic over an object
    column may come back as a string or a timestamp), so every conversion at
    this boundary is guarded rather than assumed.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result
