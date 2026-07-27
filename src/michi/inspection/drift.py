"""Comparing two datasets, or a dataset against a profile it should still match.

Design Principles
-----------------
- **The profile artifact was always half of this.** ``inspect`` has written
  schemas, distributions, and hashes since v0.1. Comparing two of them needs
  no new measurement — which is why a drift check belongs here rather than in
  a monitoring service michi would have to run.
- **Findings, not a verdict.** A shifted column is reported with its size and
  direction. Whether the shift matters depends on what the column feeds and
  what the model is for, and michi knows neither.
- **Schema changes outrank distribution changes.** A column that disappeared
  breaks the model today; a mean that moved 3% might never matter. Severity
  follows that order rather than following effect size.
- **Comparable or silent.** Two columns of different kinds are reported as a
  type change and not compared numerically, because the distance between a
  string and a float is not a number anyone should act on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from michi.core.artifacts import ColumnKind, Finding, Severity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from michi.core.artifacts import ColumnProfile, DatasetProfile

__all__ = ["DriftReport", "compare_profiles"]

# A shift smaller than this is noise on any realistic sample, and reporting it
# would bury the columns that actually moved.
_MEAN_SHIFT = 0.10
_MISSING_SHIFT = 0.05
_CARDINALITY_SHIFT = 0.50
_CATEGORY_SHARE = 0.15


class DriftReport:
    """What changed between a baseline profile and a current one.

    Attributes
    ----------
    findings
        Everything that moved, most severe first.
    baseline_rows, current_rows
        Row counts, for context in the summary.
    added, removed
        Columns present on only one side.
    """

    __slots__ = ("added", "baseline_rows", "current_rows", "findings", "removed")

    def __init__(
        self,
        findings: tuple[Finding, ...],
        *,
        baseline_rows: int,
        current_rows: int,
        added: tuple[str, ...],
        removed: tuple[str, ...],
    ) -> None:
        self.findings = findings
        self.baseline_rows = baseline_rows
        self.current_rows = current_rows
        self.added = added
        self.removed = removed

    @property
    def worst(self) -> Severity | None:
        """The highest severity present, or ``None`` when nothing moved."""
        return self.findings[0].severity if self.findings else None

    def to_dict(self) -> dict[str, object]:
        """Serialise for `--json`, in the shape a profile's findings take."""
        return {
            "baseline_rows": self.baseline_rows,
            "current_rows": self.current_rows,
            "added_columns": list(self.added),
            "removed_columns": list(self.removed),
            "findings": [item.to_dict() for item in self.findings],
        }


def compare_profiles(baseline: DatasetProfile, current: DatasetProfile) -> DriftReport:
    """Report what changed between two profiles of the same data source.

    Parameters
    ----------
    baseline
        What the data used to look like — typically a committed profile.
    current
        What it looks like now.

    Returns
    -------
    DriftReport
        Findings ordered most severe first.
    """
    before = {column.name: column for column in baseline.columns}
    after = {column.name: column for column in current.columns}

    findings: list[Finding] = []
    removed = tuple(name for name in before if name not in after)
    added = tuple(name for name in after if name not in before)

    findings.extend(_schema_findings(removed, added))
    for name in before:
        if name in after:
            findings.extend(_column_findings(before[name], after[name]))
    findings.extend(_shape_findings(baseline, current))

    order = {Severity.HIGH: 0, Severity.WARN: 1, Severity.INFO: 2}
    findings.sort(key=lambda item: order[item.severity])
    return DriftReport(
        tuple(findings),
        baseline_rows=baseline.n_rows,
        current_rows=current.n_rows,
        added=added,
        removed=removed,
    )


def _schema_findings(removed: tuple[str, ...], added: tuple[str, ...]) -> list[Finding]:
    """A column that vanished breaks a model today, whatever its distribution."""
    findings: list[Finding] = []
    if removed:
        findings.append(
            Finding(
                kind="column-removed",
                severity=Severity.HIGH,
                columns=removed,
                summary=(
                    f"{len(removed)} column(s) present in the baseline are gone: "
                    f"{', '.join(removed[:5])}"
                ),
                metrics={"count": len(removed)},
            )
        )
    if added:
        # New columns cannot break a fitted model — it never asked for them —
        # so this is information rather than a warning.
        findings.append(
            Finding(
                kind="column-added",
                severity=Severity.INFO,
                columns=added,
                summary=(f"{len(added)} new column(s): {', '.join(added[:5])}"),
                metrics={"count": len(added)},
            )
        )
    return findings


def _column_findings(before: ColumnProfile, after: ColumnProfile) -> list[Finding]:
    """Everything that moved within one column."""
    findings: list[Finding] = []

    if before.kind is not after.kind:
        # Two kinds are not comparable numerically, so this replaces the
        # distribution checks rather than adding to them.
        return [
            Finding(
                kind="type-changed",
                severity=Severity.HIGH,
                columns=(after.name,),
                summary=(
                    f"{after.name} was {before.kind.value}, is now {after.kind.value}"
                ),
                metrics={"from": before.kind.value, "to": after.kind.value},
            )
        ]

    missing_shift = after.missing_pct - before.missing_pct
    if abs(missing_shift) >= _MISSING_SHIFT * 100:
        findings.append(
            Finding(
                kind="missingness-changed",
                severity=Severity.WARN if missing_shift > 0 else Severity.INFO,
                columns=(after.name,),
                summary=(
                    f"{after.name} is {after.missing_pct:.1f}% missing, was "
                    f"{before.missing_pct:.1f}%"
                ),
                metrics={
                    "before": round(before.missing_pct, 3),
                    "after": round(after.missing_pct, 3),
                },
            )
        )

    if before.kind is ColumnKind.NUMERIC:
        findings.extend(_numeric_findings(before, after))
    elif before.kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
        findings.extend(_categorical_findings(before, after))
    return findings


def _numeric_findings(before: ColumnProfile, after: ColumnProfile) -> list[Finding]:
    """A numeric column's centre and spread, compared in its own units."""
    old_mean = before.stats.get("mean")
    new_mean = after.stats.get("mean")
    old_std = before.stats.get("std")
    if old_mean is None or new_mean is None:
        return []

    # Measured in baseline standard deviations, so the threshold means the
    # same thing for a column in dollars and one in millimetres.
    scale = float(old_std) if old_std else abs(float(old_mean)) or 1.0
    shift = (float(new_mean) - float(old_mean)) / scale
    if abs(shift) < _MEAN_SHIFT:
        return []
    return [
        Finding(
            kind="distribution-shifted",
            severity=Severity.WARN if abs(shift) >= 0.5 else Severity.INFO,
            columns=(after.name,),
            summary=(
                f"{after.name} mean moved {shift:+.2f} baseline SD "
                f"({float(old_mean):.4g} → {float(new_mean):.4g})"
            ),
            metrics={
                "shift_sd": round(shift, 4),
                "before": round(float(old_mean), 6),
                "after": round(float(new_mean), 6),
            },
        )
    ]


def _categorical_findings(before: ColumnProfile, after: ColumnProfile) -> list[Finding]:
    """New levels, vanished levels, and a majority that changed hands."""
    findings: list[Finding] = []

    if before.unique:
        ratio = after.unique / before.unique
        if abs(ratio - 1.0) >= _CARDINALITY_SHIFT:
            findings.append(
                Finding(
                    kind="cardinality-changed",
                    severity=Severity.WARN,
                    columns=(after.name,),
                    summary=(
                        f"{after.name} has {after.unique:,} distinct values, "
                        f"was {before.unique:,}"
                    ),
                    metrics={"before": before.unique, "after": after.unique},
                )
            )

    old_top = {str(value): count for value, count in before.top_values}
    new_top = {str(value): count for value, count in after.top_values}
    unseen = [name for name in new_top if name not in old_top]
    if unseen:
        # A category the training data never contained is the categorical
        # equivalent of a missing column: an encoder has no slot for it.
        findings.append(
            Finding(
                kind="new-categories",
                severity=Severity.WARN,
                columns=(after.name,),
                summary=(
                    f"{after.name} has value(s) absent from the baseline: "
                    f"{', '.join(unseen[:5])}"
                ),
                metrics={"count": len(unseen)},
            )
        )

    old_share = _share(before)
    new_share = _share(after)
    if old_share and new_share and abs(new_share[1] - old_share[1]) >= _CATEGORY_SHARE:
        findings.append(
            Finding(
                kind="category-share-changed",
                severity=Severity.INFO,
                columns=(after.name,),
                summary=(
                    f"{after.name}: {new_share[0]!r} is now "
                    f"{new_share[1]:.0%} of rows, was {old_share[1]:.0%}"
                ),
                metrics={
                    "value": new_share[0],
                    "before": round(old_share[1], 4),
                    "after": round(new_share[1], 4),
                },
            )
        )
    return findings


def _share(column: ColumnProfile) -> tuple[str, float] | None:
    """The most common value and the fraction of non-missing rows it holds."""
    if not column.top_values or not column.count:
        return None
    value, count = column.top_values[0]
    return str(value), count / column.count


def _shape_findings(baseline: DatasetProfile, current: DatasetProfile) -> list[Finding]:
    """A row count that collapsed usually means a broken export, not drift."""
    if not baseline.n_rows:
        return []
    ratio = current.n_rows / baseline.n_rows
    if 0.5 <= ratio <= 2.0:
        return []
    return [
        Finding(
            kind="row-count-changed",
            severity=Severity.WARN,
            columns=(),
            summary=(
                f"{current.n_rows:,} rows, against {baseline.n_rows:,} in the "
                f"baseline ({ratio:.2f}×)"
            ),
            metrics={"before": baseline.n_rows, "after": current.n_rows},
        )
    ]
