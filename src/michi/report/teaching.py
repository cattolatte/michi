"""Explaining the numbers a benchmark just produced, not statistics in general.

Design Principles
-----------------
- **About *these* numbers.** A textbook paragraph on confidence intervals is a
  textbook's job. What michi can say that a textbook cannot is why *this*
  interval is *this* wide, and why *these two* models could not be separated
  — using the folds that were actually run.
- **Description, never advice.** Every sentence here states a fact about the
  result or an arithmetic consequence of it. "You would need roughly eight
  times the rows to separate these two" is a calculation. "You should collect
  more data" is a judgement, and not michi's to make.
- **Arithmetic is labelled as arithmetic.** The sample-size figures are
  back-of-envelope: they assume the observed difference is real and that
  error shrinks with the square root of the sample. Both are stated where
  the number is given, because an unqualified number gets quoted.
- **Silence over speculation.** A section that has nothing specific to say
  about this result is omitted rather than padded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from michi.bench import BenchResult

__all__ = ["teaching_notes"]

_TIE_TARGET_T = 2.0
"""Roughly the t-statistic a difference needs to clear at these fold counts."""


def teaching_notes(result: BenchResult) -> tuple[str, ...]:
    """Explain this benchmark's own numbers, one paragraph per point.

    Parameters
    ----------
    result
        A completed benchmark.

    Returns
    -------
    tuple of str
        Paragraphs, already wrapped-agnostic prose. Empty when the result has
        nothing specific to say — a single successful model has no comparison
        to explain.
    """
    notes: list[str] = []
    leader = result.leader
    if leader is None:
        return ()

    notes.extend(_baseline_note(result, leader))
    notes.extend(_spread_note(result, leader))
    notes.extend(_tie_notes(result, leader))
    return tuple(notes)


def _baseline_note(result: BenchResult, leader: object) -> list[str]:
    """What the dummy baseline says about whether any of this matters."""
    dummy = next((item for item in result.results if item.name == "dummy"), None)
    if dummy is None or dummy.failed is not None:
        return []

    best = float(leader.primary.value)  # type: ignore[attr-defined]
    floor = float(dummy.primary.value)
    gap = best - floor
    name = leader.name  # type: ignore[attr-defined]

    if gap <= 0:
        return [
            f"No model beat the dummy baseline. The baseline scores "
            f"{floor:.4g} on {result.primary_metric} by ignoring the features "
            f"entirely, and {name}, the best of the rest, scores {best:.4g}. "
            "Whatever these features carry, none of these models found it."
        ]
    return [
        f"The dummy baseline scores {floor:.4g} without looking at a single "
        f"feature — it is what you get for free. {name} scores {best:.4g}, so "
        f"the features are worth {gap:.4g} of {result.primary_metric} here. "
        "That gap, not the headline score, is what the modelling bought."
    ]


def _spread_note(result: BenchResult, leader: object) -> list[str]:
    """Why the leader's interval is as wide as it is."""
    scores = tuple(leader.fold_scores)  # type: ignore[attr-defined]
    if len(scores) < 2:
        return []

    low, high = min(scores), max(scores)
    if high == low:
        return []

    name = leader.name  # type: ignore[attr-defined]
    held_out = result.n_rows // result.folds if result.folds else 0
    return [
        f"{name} scored between {low:.4g} and {high:.4g} across the "
        f"{result.folds} folds — a spread of {high - low:.4g} on the same data "
        f"and the same settings, differing only in which {held_out:,} rows "
        "were held out. That spread is the reason the interval has the width "
        "it does; it is measuring how much of the score is the fold rather "
        "than the model."
    ]


def _tie_notes(result: BenchResult, leader: object) -> list[str]:
    """Why the models that tied could not be separated, in rows."""
    import numpy as np

    by_name = {item.name: item for item in result.results}
    leader_scores = np.asarray(list(leader.fold_scores))  # type: ignore[attr-defined]
    notes: list[str] = []

    tied = [
        comparison
        for comparison in result.comparisons
        if comparison.model != comparison.leader
        and not comparison.significant
        and comparison.model != "dummy"
    ]
    if not tied:
        return notes

    for comparison in tied:
        challenger = by_name.get(comparison.model)
        if challenger is None or challenger.failed is not None:
            continue
        differences = leader_scores - np.asarray(list(challenger.fold_scores))
        if differences.shape[0] < 2:
            continue
        mean = float(np.mean(differences))
        spread = float(np.std(differences, ddof=1))
        if spread <= 0 or mean == 0:
            continue

        notes.append(
            f"{comparison.model} scored {abs(mean):.4g} "
            f"{'below' if mean > 0 else 'above'} {comparison.leader} on average, "
            f"but the two swapped places by ±{spread:.4g} from fold to fold. "
            "The gap is smaller than the disagreement about the gap, which is "
            f"what {comparison.formatted_p} is reporting."
        )

        needed = _rows_to_separate(result.n_rows, mean, spread, result.folds)
        if needed is not None:
            notes.append(
                f"Back-of-envelope: separating them at this effect size would "
                f"take roughly {needed:,} rows, against the {result.n_rows:,} "
                "here. That assumes the difference you measured is real and "
                "that error falls with the square root of the sample — both "
                "optimistic, so read it as an order of magnitude, not a target."
            )
        break  # One worked example is instructive; five is a wall of text.
    return notes


def _rows_to_separate(
    n_rows: int, mean: float, spread: float, folds: int
) -> int | None:
    """Rows at which a difference this size would clear the significance bar.

    The corrected resampled statistic scales the standard error by a factor
    that does not depend on the sample size, so the sample-size ratio needed
    is the square of how much the error has to shrink.
    """
    if n_rows <= 0 or spread <= 0 or mean == 0:
        return None
    correction = (1.0 / folds) + (1.0 / (folds - 1)) if folds > 1 else 1.0
    statistic = abs(mean) / (spread * correction**0.5)
    if statistic <= 0:
        return None
    growth = (_TIE_TARGET_T / statistic) ** 2
    if growth <= 1.0:
        return None
    if growth > 1e4:
        # Beyond this the honest answer is "not with more rows of this kind",
        # and a number with five digits of false precision says the opposite.
        return None
    return int(round(n_rows * growth, -2)) or None
