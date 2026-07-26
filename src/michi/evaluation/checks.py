"""Checks that turn evaluation numbers into findings.

Design Principles
-----------------
- **The checks encode what reviewers actually ask.** Does it beat a trivial
  baseline? Is the score too good to be true? Does it collapse onto one class?
  Are the probabilities meaningful? Does it fail a particular subgroup?
- **Findings, not verdicts.** Each check states what the numbers show and
  leaves the interpretation — and the fix — to the user, reusing the same
  ``Finding`` type and explanation machinery as ``michi inspect``.
- **Silence is meaningful.** A check that does not fire says nothing, so a
  clean evaluation stays quiet rather than padding the output.
"""

from __future__ import annotations

from typing import Any, Final

from michi.core.artifacts import Finding, Severity
from michi.core.manifest import Metric

__all__ = ["evaluation_checks"]

_PERFECT_SCORE: Final = 0.999
_BASELINE_MARGIN: Final = 0.01
_HIGH_CALIBRATION_ERROR: Final = 0.1
_SLICE_GAP: Final = 0.15
_SMALL_EVALUATION_ROWS: Final = 100


def evaluation_checks(
    *,
    task: str,
    metrics: tuple[Metric, ...],
    baselines: dict[str, tuple[Metric, ...]],
    slices: tuple[Any, ...],
    details: dict[str, Any],
    n_rows: int,
) -> tuple[Finding, ...]:
    """Run every evaluation check and return the findings it raised."""
    findings: list[Finding] = []
    findings.extend(_baseline_check(metrics, baselines))
    findings.extend(_perfect_score_check(task, metrics))
    findings.extend(_wide_interval_check(metrics))
    findings.extend(_class_collapse_check(details))
    findings.extend(_calibration_check(details))
    findings.extend(_slice_gap_check(slices))
    findings.extend(_sample_size_check(n_rows))
    return tuple(sorted(findings, key=lambda f: (f.severity.rank, f.kind, f.columns)))


def _primary(metrics: tuple[Metric, ...]) -> Metric | None:
    for metric in metrics:
        if metric.value == metric.value:  # not NaN
            return metric
    return None


def _baseline_check(
    metrics: tuple[Metric, ...], baselines: dict[str, tuple[Metric, ...]]
) -> list[Finding]:
    """Compare the model against trivial models on the same rows."""
    primary = _primary(metrics)
    if primary is None or not baselines:
        return []

    comparisons: list[tuple[str, float]] = []
    for name, baseline_metrics in baselines.items():
        for metric in baseline_metrics:
            if metric.name == primary.name and metric.value == metric.value:
                comparisons.append((name, metric.value))
                break
    if not comparisons:
        return []

    if primary.greater_is_better:
        best_name, best_value = max(comparisons, key=lambda item: item[1])
        beaten = primary.value > best_value + _BASELINE_MARGIN
        margin = primary.value - best_value
    else:
        best_name, best_value = min(comparisons, key=lambda item: item[1])
        beaten = primary.value < best_value - _BASELINE_MARGIN
        margin = best_value - primary.value

    if beaten:
        return [
            Finding(
                kind="beats-baseline",
                severity=Severity.INFO,
                columns=(),
                summary=(
                    f"{primary.name} {primary.value:.4g} beats the {best_name} "
                    f"baseline ({best_value:.4g}) by {margin:.4g}"
                ),
                metrics={
                    "metric": primary.name,
                    "model": round(primary.value, 6),
                    "baseline": round(best_value, 6),
                },
            )
        ]
    return [
        Finding(
            kind="below-baseline",
            severity=Severity.HIGH,
            columns=(),
            summary=(
                f"{primary.name} {primary.value:.4g} does not beat the "
                f"{best_name} baseline ({best_value:.4g})"
            ),
            metrics={
                "metric": primary.name,
                "model": round(primary.value, 6),
                "baseline": round(best_value, 6),
            },
        )
    ]


def _perfect_score_check(task: str, metrics: tuple[Metric, ...]) -> list[Finding]:
    """A near-perfect score is evidence of leakage far more often than skill."""
    primary = _primary(metrics)
    if primary is None:
        return []
    perfect = (
        primary.value >= _PERFECT_SCORE
        if task == "classification"
        else primary.name == "r2" and primary.value >= _PERFECT_SCORE
    )
    if not perfect:
        return []
    return [
        Finding(
            kind="suspiciously-perfect",
            severity=Severity.HIGH,
            columns=(),
            summary=f"{primary.name} is {primary.value:.4g} — near perfect",
            metrics={"metric": primary.name, "value": round(primary.value, 6)},
        )
    ]


def _wide_interval_check(metrics: tuple[Metric, ...]) -> list[Finding]:
    """A wide interval means the score cannot support fine comparisons."""
    primary = _primary(metrics)
    if primary is None or not primary.has_interval:
        return []
    assert primary.ci_low is not None and primary.ci_high is not None
    width = primary.ci_high - primary.ci_low
    if width < 0.1:
        return []
    return [
        Finding(
            kind="wide-interval",
            severity=Severity.WARN,
            columns=(),
            summary=(
                f"{primary.name} interval spans {width:.3g} "
                f"({primary.ci_low:.4g} to {primary.ci_high:.4g})"
            ),
            metrics={"metric": primary.name, "width": round(width, 5)},
        )
    ]


def _class_collapse_check(details: dict[str, Any]) -> list[Finding]:
    """A model predicting one class only has learned the prior, not the task."""
    confusion = details.get("confusion")
    classes = details.get("classes")
    if not confusion or not classes or len(classes) < 2:
        return []

    predicted_totals = [
        sum(row[index] for row in confusion) for index in range(len(classes))
    ]
    used = [index for index, total in enumerate(predicted_totals) if total > 0]
    if len(used) > 1:
        return []
    return [
        Finding(
            kind="single-class-predictions",
            severity=Severity.HIGH,
            columns=(),
            summary=(
                f"every prediction is {classes[used[0]]!r}; the model never "
                f"predicts the other {len(classes) - 1} class(es)"
            ),
            metrics={"predicted_class": str(classes[used[0]])},
        )
    ]


def _calibration_check(details: dict[str, Any]) -> list[Finding]:
    """Probabilities that do not match observed frequencies mislead."""
    error = details.get("ece")
    if error is None or float(error) < _HIGH_CALIBRATION_ERROR:
        return []
    return [
        Finding(
            kind="miscalibrated",
            severity=Severity.WARN,
            columns=(),
            summary=(
                f"predicted probabilities are off by {float(error):.1%} on "
                "average (expected calibration error)"
            ),
            metrics={"ece": round(float(error), 4)},
        )
    ]


def _slice_gap_check(slices: tuple[Any, ...]) -> list[Finding]:
    """A large gap between subgroups is invisible in an aggregate score."""
    findings: list[Finding] = []
    by_column: dict[str, list[Any]] = {}
    for item in slices:
        by_column.setdefault(item.column, []).append(item)

    for column, group in by_column.items():
        scores = [item.score for item in group if item.score == item.score]
        if len(scores) < 2:
            continue
        worst = min(group, key=lambda item: item.score)
        best = max(group, key=lambda item: item.score)
        gap = best.score - worst.score
        if gap < _SLICE_GAP:
            continue
        findings.append(
            Finding(
                kind="slice-gap",
                severity=Severity.WARN,
                columns=(column,),
                summary=(
                    f"{worst.metric} ranges from {worst.score:.3g} "
                    f"({column}={worst.value}) to {best.score:.3g} "
                    f"({column}={best.value})"
                ),
                metrics={
                    "gap": round(gap, 4),
                    "worst_group": str(worst.value),
                    "best_group": str(best.value),
                },
            )
        )
    return findings


def _sample_size_check(n_rows: int) -> list[Finding]:
    """Few evaluation rows make every metric a wide guess."""
    if n_rows >= _SMALL_EVALUATION_ROWS:
        return []
    return [
        Finding(
            kind="small-evaluation-set",
            severity=Severity.WARN,
            columns=(),
            summary=f"only {n_rows} rows were evaluated",
            metrics={"rows": n_rows},
        )
    ]
