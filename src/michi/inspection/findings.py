"""Detecting findings — observations a user may want to act on.

Design Principles
-----------------
- **Findings are observations, not advice.** Each one states a measurable fact
  ("77.1% of values are missing"). What to do about it is the user's call, and
  the option menus live in the explanation layer.
- **Thresholds are explicit and named.** Every cutoff lives in
  :class:`Thresholds` with a documented default, so users can see exactly why
  something was flagged instead of guessing at hidden magic numbers.
- **Detectors are independent functions.** Each takes the frame and column
  profiles and returns findings; adding one never requires touching another.
- **Bounded cost.** Pairwise work (correlation, duplicate columns) is capped
  on wide datasets so profiling never becomes the slow step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from michi.core.artifacts import ColumnKind, ColumnProfile, Finding, Severity

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["Thresholds", "detect_findings"]

_MAX_PAIRWISE_COLUMNS: Final = 200
_PARSE_SAMPLE: Final = 500

# A perfect partition of the target is only evidence of leakage when the
# groups are substantial. Any column with roughly one row per distinct value
# partitions any target trivially — that is an identifier, not a leak — and a
# mostly-absent column cannot be a meaningful predictor of anything.
_MIN_PARTITION_GROUP_ROWS: Final = 3.0
_MIN_PARTITION_COVERAGE: Final = 0.5

# Punctuation that decorates numbers in exported data.
_NUMERIC_NOISE: Final = r"[,\s$€£¥%_]"

# A date needs separators or a month name; a bare integer is not a date, even
# though pandas will cheerfully read one as a year.
_DATE_SHAPE: Final = (
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\d{4}-\d{2}"
    r"|\d{1,2}:\d{2}"
    r"|(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
)


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Named cutoffs that decide when an observation becomes a finding.

    Defaults are conventional rather than clever; they are exposed as a value
    object so that they can be surfaced in reports and, later, configured.

    Examples
    --------
    >>> Thresholds().high_missing_pct
    50.0
    """

    high_missing_pct: float = 50.0
    warn_missing_pct: float = 5.0
    high_cardinality_ratio: float = 0.5
    high_cardinality_min: int = 50
    skew_info: float = 2.0
    skew_warn: float = 5.0
    outlier_info_pct: float = 1.0
    outlier_warn_pct: float = 5.0
    correlation: float = 0.95
    leakage_correlation: float = 0.98
    imbalance_warn_pct: float = 20.0
    imbalance_high_pct: float = 5.0
    tiny_dataset_rows: int = 50
    text_parse_ratio: float = 0.95


def detect_findings(
    frame: pd.DataFrame,
    *,
    columns: tuple[ColumnProfile, ...],
    target: str | None,
    duplicate_rows: int,
    thresholds: Thresholds,
) -> tuple[Finding, ...]:
    """Run every detector and return the findings, most severe first."""
    findings: list[Finding] = []
    findings.extend(_dataset_shape(frame, duplicate_rows, thresholds))
    findings.extend(_column_quality(columns, thresholds))
    findings.extend(_distribution(columns, thresholds))
    findings.extend(_duplicate_columns(frame))
    findings.extend(_correlated_pairs(frame, columns, thresholds))
    findings.extend(_text_encoded_values(frame, columns, thresholds))
    if target is not None:
        findings.extend(_target_findings(frame, columns, target, thresholds))
    return tuple(sorted(findings, key=lambda f: (f.severity.rank, f.kind, f.columns)))


# --- dataset-level ---------------------------------------------------------


def _dataset_shape(
    frame: pd.DataFrame, duplicate_rows: int, thresholds: Thresholds
) -> list[Finding]:
    findings: list[Finding] = []
    n_rows = int(frame.shape[0])

    if n_rows < thresholds.tiny_dataset_rows:
        findings.append(
            Finding(
                kind="tiny-dataset",
                severity=Severity.WARN,
                columns=(),
                summary=f"only {n_rows} rows",
                metrics={"rows": n_rows},
            )
        )
    if duplicate_rows:
        pct = 100.0 * duplicate_rows / n_rows if n_rows else 0.0
        findings.append(
            Finding(
                kind="duplicate-rows",
                severity=Severity.WARN if pct >= 1.0 else Severity.INFO,
                columns=(),
                summary=f"{duplicate_rows} duplicate rows ({pct:.1f}%)",
                metrics={"count": duplicate_rows, "pct": round(pct, 3)},
            )
        )
    return findings


# --- per-column quality ----------------------------------------------------


def _column_quality(
    columns: tuple[ColumnProfile, ...], thresholds: Thresholds
) -> list[Finding]:
    findings: list[Finding] = []
    for column in columns:
        if column.kind is ColumnKind.EMPTY:
            findings.append(
                Finding(
                    kind="empty-column",
                    severity=Severity.HIGH,
                    columns=(column.name,),
                    summary="every value is missing",
                    metrics={"rows": column.total},
                )
            )
            continue

        if column.unique == 1:
            value = column.top_values[0][0] if column.top_values else "a single value"
            findings.append(
                Finding(
                    kind="constant-column",
                    severity=Severity.HIGH,
                    columns=(column.name,),
                    summary=f"only one distinct value ({value})",
                    metrics={"value": value},
                )
            )

        if column.missing:
            pct = column.missing_pct
            severity = (
                Severity.HIGH
                if pct >= thresholds.high_missing_pct
                else Severity.WARN
                if pct >= thresholds.warn_missing_pct
                else Severity.INFO
            )
            findings.append(
                Finding(
                    kind="high-missing" if severity is Severity.HIGH else "missing",
                    severity=severity,
                    columns=(column.name,),
                    summary=f"{pct:.1f}% missing ({column.missing} of {column.total})",
                    metrics={"pct": round(pct, 3), "count": column.missing},
                )
            )

        if (
            column.kind in {ColumnKind.CATEGORICAL, ColumnKind.TEXT}
            and column.count
            and column.unique == column.count
            and column.count > 10
            # A mostly-absent column is reported as missing, not as an
            # identifier: distinctness among a handful of surviving values
            # says nothing useful.
            and column.missing_pct < thresholds.high_missing_pct
        ):
            findings.append(
                Finding(
                    kind="identifier-like",
                    severity=Severity.INFO,
                    columns=(column.name,),
                    summary="every value is distinct (looks like an identifier)",
                    metrics={"unique": column.unique},
                )
            )
        elif (
            column.kind is ColumnKind.CATEGORICAL
            and column.count
            and column.unique >= thresholds.high_cardinality_min
            and column.unique / column.count >= thresholds.high_cardinality_ratio
        ):
            findings.append(
                Finding(
                    kind="high-cardinality",
                    severity=Severity.WARN,
                    columns=(column.name,),
                    summary=(
                        f"{column.unique} distinct values across {column.count} rows"
                    ),
                    metrics={
                        "unique": column.unique,
                        "ratio": round(column.unique / column.count, 3),
                    },
                )
            )
    return findings


# --- distributions ---------------------------------------------------------


def _distribution(
    columns: tuple[ColumnProfile, ...], thresholds: Thresholds
) -> list[Finding]:
    findings: list[Finding] = []
    for column in columns:
        if column.kind is not ColumnKind.NUMERIC or not column.stats:
            continue

        skew = column.stats.get("skew")
        if skew is not None and abs(skew) >= thresholds.skew_info:
            tail = "right" if skew > 0 else "left"
            findings.append(
                Finding(
                    kind="high-skew",
                    severity=(
                        Severity.WARN
                        if abs(skew) >= thresholds.skew_warn
                        else Severity.INFO
                    ),
                    columns=(column.name,),
                    summary=f"skew {skew:.2f} ({tail}-tailed)",
                    metrics={"skew": round(skew, 4)},
                )
            )

        outliers = column.stats.get("outliers", 0.0)
        if outliers and column.count:
            pct = 100.0 * outliers / column.count
            if pct >= thresholds.outlier_info_pct:
                findings.append(
                    Finding(
                        kind="outliers",
                        severity=(
                            Severity.WARN
                            if pct >= thresholds.outlier_warn_pct
                            else Severity.INFO
                        ),
                        columns=(column.name,),
                        summary=(f"{int(outliers)} values beyond 1.5×IQR ({pct:.1f}%)"),
                        metrics={"count": int(outliers), "pct": round(pct, 3)},
                    )
                )
    return findings


# --- cross-column ----------------------------------------------------------


def _duplicate_columns(frame: pd.DataFrame) -> list[Finding]:
    """Find columns holding identical values, compared by content hash."""
    import pandas as pd

    if frame.shape[1] > _MAX_PAIRWISE_COLUMNS:
        return []

    signatures: dict[str, list[str]] = {}
    for name in frame.columns:
        try:
            digest = pd.util.hash_pandas_object(frame[name], index=False).sum()
        except TypeError:
            digest = pd.util.hash_pandas_object(
                frame[name].astype(str), index=False
            ).sum()
        signatures.setdefault(str(digest), []).append(str(name))

    findings: list[Finding] = []
    for group in signatures.values():
        if len(group) < 2:
            continue
        first = frame[group[0]]
        identical = [name for name in group[1:] if frame[name].equals(first)]
        if identical:
            members = tuple(sorted([group[0], *identical]))
            findings.append(
                Finding(
                    kind="duplicate-columns",
                    severity=Severity.WARN,
                    columns=members,
                    summary=f"identical values in {', '.join(members)}",
                    metrics={"count": len(members)},
                )
            )
    return findings


def _correlated_pairs(
    frame: pd.DataFrame,
    columns: tuple[ColumnProfile, ...],
    thresholds: Thresholds,
) -> list[Finding]:
    """Find numeric column pairs whose absolute correlation is extreme."""
    numeric = [
        column.name
        for column in columns
        if column.kind is ColumnKind.NUMERIC and column.unique > 1
    ]
    if len(numeric) < 2 or len(numeric) > _MAX_PAIRWISE_COLUMNS:
        return []

    try:
        matrix = frame[numeric].corr(numeric_only=True)
    except (TypeError, ValueError):
        return []

    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for left in matrix.columns:
        for right in matrix.columns:
            if left == right:
                continue
            pair = tuple(sorted((str(left), str(right))))
            if pair in seen:
                continue
            value = matrix.at[left, right]
            if value is None or not isinstance(value, float) or value != value:
                continue
            if abs(value) >= thresholds.correlation:
                seen.add((pair[0], pair[1]))
                findings.append(
                    Finding(
                        kind="highly-correlated",
                        severity=Severity.WARN,
                        columns=pair,
                        summary=f"correlation {value:+.3f}",
                        metrics={"correlation": round(float(value), 4)},
                    )
                )
    return findings


def _text_encoded_values(
    frame: pd.DataFrame,
    columns: tuple[ColumnProfile, ...],
    thresholds: Thresholds,
) -> list[Finding]:
    """Find text columns whose values are really numbers, dates, or mixed."""

    findings: list[Finding] = []
    for column in columns:
        if column.kind not in {ColumnKind.CATEGORICAL, ColumnKind.TEXT}:
            continue
        series = frame[column.name].dropna()
        if series.empty:
            continue
        sample = series.head(_PARSE_SAMPLE)

        numeric_ratio = _numeric_ratio(sample)
        if numeric_ratio >= thresholds.text_parse_ratio:
            findings.append(
                Finding(
                    kind="numeric-stored-as-text",
                    severity=Severity.WARN,
                    columns=(column.name,),
                    summary=f"{numeric_ratio:.0%} of values parse as numbers",
                    metrics={"ratio": round(numeric_ratio, 3)},
                )
            )
        elif _parses_as_datetime(sample, thresholds.text_parse_ratio):
            findings.append(
                Finding(
                    kind="datetime-stored-as-text",
                    severity=Severity.WARN,
                    columns=(column.name,),
                    summary="values parse as dates but are stored as text",
                    metrics={"sampled": int(sample.shape[0])},
                )
            )

        type_names = {type(value).__name__ for value in sample}
        if len(type_names) > 1:
            findings.append(
                Finding(
                    kind="mixed-types",
                    severity=Severity.WARN,
                    columns=(column.name,),
                    summary=f"mixed value types ({', '.join(sorted(type_names))})",
                    metrics={"types": ", ".join(sorted(type_names))},
                )
            )
    return findings


def _numeric_ratio(sample: pd.Series[Any]) -> float:
    """Fraction of a sample that represents a number.

    Values are tried as-is first, then with the punctuation that commonly
    surrounds numbers in exported data — thousands separators, currency
    symbols, percent signs — removed. Without the second pass, ``"1,234"``
    and ``"$9.99"`` look like categories rather than the numbers they are.
    """
    import pandas as pd

    direct = float(pd.to_numeric(sample, errors="coerce").notna().mean())
    if direct >= 1.0:
        return direct
    stripped = sample.astype(str).str.replace(_NUMERIC_NOISE, "", regex=True)
    cleaned = float(pd.to_numeric(stripped, errors="coerce").notna().mean())
    return max(direct, cleaned)


def _parses_as_datetime(sample: pd.Series[Any], ratio: float) -> bool:
    """Whether enough of a sample parses as timestamps to be worth flagging.

    A cheap shape check runs first: pandas will happily read a bare integer
    as a year, so without requiring date-like punctuation every numeric-ish
    text column would be reported as a date.
    """
    import warnings

    import pandas as pd

    text = sample.astype(str)
    if float(text.str.contains(_DATE_SHAPE, regex=True, na=False).mean()) < ratio:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return False
    return bool(parsed.notna().mean() >= ratio)


# --- target-aware ----------------------------------------------------------


def _target_findings(
    frame: pd.DataFrame,
    columns: tuple[ColumnProfile, ...],
    target: str,
    thresholds: Thresholds,
) -> list[Finding]:
    findings: list[Finding] = []
    profile = next((c for c in columns if c.name == target), None)
    if profile is None:
        return findings

    findings.extend(_class_imbalance(frame, profile, thresholds))
    findings.extend(_leakage_suspects(frame, columns, target, thresholds))
    return findings


def _class_imbalance(
    frame: pd.DataFrame, target: ColumnProfile, thresholds: Thresholds
) -> list[Finding]:
    """Flag a skewed class distribution for a classification-shaped target."""
    is_classification = target.kind in {
        ColumnKind.CATEGORICAL,
        ColumnKind.BOOLEAN,
    } or (target.kind is ColumnKind.NUMERIC and target.unique <= 20)
    if not is_classification or target.count == 0 or target.unique < 2:
        return []

    counts = frame[target.name].value_counts(normalize=True)
    minority_pct = 100.0 * float(counts.min())
    majority_pct = 100.0 * float(counts.max())
    if minority_pct >= thresholds.imbalance_warn_pct:
        return []

    severity = (
        Severity.HIGH if minority_pct < thresholds.imbalance_high_pct else Severity.WARN
    )
    return [
        Finding(
            kind="class-imbalance",
            severity=severity,
            columns=(target.name,),
            summary=(
                f"smallest class {minority_pct:.1f}% vs largest "
                f"{majority_pct:.1f}% across {target.unique} classes"
            ),
            metrics={
                "minority_pct": round(minority_pct, 3),
                "majority_pct": round(majority_pct, 3),
                "classes": target.unique,
            },
        )
    ]


def _leakage_suspects(
    frame: pd.DataFrame,
    columns: tuple[ColumnProfile, ...],
    target: str,
    thresholds: Thresholds,
) -> list[Finding]:
    """Flag features suspiciously predictive of the target.

    A feature that almost perfectly determines the label is usually a leak —
    a value recorded after the outcome — rather than a genuinely excellent
    predictor. michi only raises the suspicion; confirming it needs domain
    knowledge michi does not have.
    """
    findings: list[Finding] = []
    target_profile = next((c for c in columns if c.name == target), None)
    if target_profile is None or frame.shape[0] < 10:
        return findings

    flagged: set[str] = set()

    # A numeric target admits a correlation test against numeric features.
    if target_profile.kind is ColumnKind.NUMERIC:
        findings.extend(
            _correlation_leakage(frame, columns, target, thresholds, flagged)
        )

    # A classification-shaped target — categorical, boolean, or numeric with
    # few distinct values — additionally admits a perfect-partition test that
    # catches categorical leaks a correlation would miss entirely.
    is_classification = target_profile.kind in {
        ColumnKind.CATEGORICAL,
        ColumnKind.BOOLEAN,
    } or (target_profile.kind is ColumnKind.NUMERIC and target_profile.unique <= 20)
    if is_classification:
        findings.extend(_partition_leakage(frame, columns, target, flagged))
    return findings


def _correlation_leakage(
    frame: pd.DataFrame,
    columns: tuple[ColumnProfile, ...],
    target: str,
    thresholds: Thresholds,
    flagged: set[str],
) -> list[Finding]:
    """Numeric features almost perfectly correlated with a numeric target."""
    findings: list[Finding] = []
    numeric = [
        c.name
        for c in columns
        if c.kind is ColumnKind.NUMERIC and c.name != target and c.unique > 1
    ][:_MAX_PAIRWISE_COLUMNS]
    for name in numeric:
        try:
            value = float(frame[name].corr(frame[target]))
        except (TypeError, ValueError):
            continue
        if value == value and abs(value) >= thresholds.leakage_correlation:
            flagged.add(name)
            findings.append(
                Finding(
                    kind="leakage-suspect",
                    severity=Severity.HIGH,
                    columns=(name,),
                    summary=f"correlation {value:+.3f} with target {target!r}",
                    metrics={"correlation": round(value, 4)},
                )
            )
    return findings


def _partition_leakage(
    frame: pd.DataFrame,
    columns: tuple[ColumnProfile, ...],
    target: str,
    flagged: set[str],
) -> list[Finding]:
    """Features whose every value maps to exactly one class of the target."""
    import pandas as pd

    findings: list[Finding] = []
    for column in columns:
        if (
            column.name == target
            or column.name in flagged
            or column.kind is ColumnKind.EMPTY
        ):
            continue
        if column.unique < 2 or column.count == 0:
            continue
        if column.count / column.unique < _MIN_PARTITION_GROUP_ROWS:
            continue
        if pd.notna(frame[column.name]).mean() < _MIN_PARTITION_COVERAGE:
            continue
        try:
            grouped = frame.groupby(column.name, observed=True)[target].nunique()
        except (TypeError, ValueError):
            continue
        if grouped.empty or not bool((grouped <= 1).all()):
            continue
        flagged.add(column.name)
        coverage = float(pd.notna(frame[column.name]).mean())
        findings.append(
            Finding(
                kind="leakage-suspect",
                severity=Severity.HIGH,
                columns=(column.name,),
                summary=(
                    f"each of its {column.unique} values maps to exactly one "
                    f"{target!r} class"
                ),
                metrics={"groups": column.unique, "coverage": round(coverage, 3)},
            )
        )
    return findings
