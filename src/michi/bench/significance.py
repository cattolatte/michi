"""Deciding whether one model actually beat another.

Design Principles
-----------------
- **Cross-validation folds are not independent.** Their training sets overlap
  heavily, so a naive paired t-test over folds badly understates the variance
  and calls noise significant. michi uses the corrected resampled t-test of
  Nadeau and Bengio (2003), which inflates the variance by the train/test
  ratio to account for that overlap.
- **Many comparisons need a correction.** Comparing every model against the
  leader means many tests, so p-values are adjusted with the Holm–Bonferroni
  step-down procedure — uniformly more powerful than plain Bonferroni and just
  as safe.
- **The plain-language verdict is the product.** "B scores higher than A, but
  not significantly" is the sentence that changes what a user does, so michi
  writes it out rather than leaving a p-value to be interpreted.
- Significance is never a ranking: a test says whether a difference is
  distinguishable from noise, not which model to use.

References
----------
Nadeau, C. and Bengio, Y. (2003). Inference for the Generalization Error.
Machine Learning 52(3), 239–281.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

__all__ = ["Comparison", "compare_to_leader", "corrected_paired_t_test"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


@dataclass(frozen=True, slots=True)
class Comparison:
    """The outcome of comparing one model against the leading one."""

    model: str
    leader: str
    difference: float
    p_value: float
    adjusted_p: float
    significant: bool

    @property
    def verdict(self) -> str:
        """A sentence a reader can act on without knowing what a p-value is."""
        if self.model == self.leader:
            return "leader"
        if self.significant:
            return f"worse than {self.leader} ({self.formatted_p})"
        return f"not distinguishable from {self.leader} ({self.formatted_p})"

    @property
    def formatted_p(self) -> str:
        """The adjusted p-value, reported as a bound when it underflows.

        A difference that is identical on every fold sends the statistic to
        infinity and the p-value to zero. Printing ``p=0`` reads as a bug
        rather than as the certainty it represents, so very small values are
        shown as a bound.
        """
        if self.adjusted_p < 1e-4:
            return "p<0.0001"
        return f"p={self.adjusted_p:.3g}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in a run manifest."""
        return {
            "model": self.model,
            "leader": self.leader,
            "difference": self.difference,
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
            "significant": self.significant,
        }


def corrected_paired_t_test(
    first: np.ndarray[Any, Any],
    second: np.ndarray[Any, Any],
    *,
    train_size: int,
    test_size: int,
) -> float:
    """Return the p-value for two models' per-fold scores.

    Parameters
    ----------
    first, second
        Per-fold scores of the two models, paired by fold.
    train_size, test_size
        Rows in each training and test fold, used for the variance correction.

    Returns
    -------
    float
        Two-sided p-value. ``1.0`` when the difference is exactly zero or the
        variance cannot be estimated.

    Notes
    -----
    The correction multiplies the usual variance of the mean difference by
    ``1/k + test_size/train_size``. Without it, overlapping training sets make
    repeated folds look like independent evidence and almost any difference
    appears significant.

    Examples
    --------
    >>> import numpy as np
    >>> identical = np.array([0.8, 0.82, 0.79])
    >>> corrected_paired_t_test(
    ...     identical, identical, train_size=80, test_size=20
    ... )
    1.0
    """
    import numpy as np
    from scipy import stats

    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    folds = differences.shape[0]
    if folds < 2:
        return 1.0

    mean = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1))
    if mean == 0.0:
        return 1.0
    if variance <= 0.0:
        # Every fold shows exactly the same non-zero difference. That is
        # perfectly consistent evidence, and the t-statistic diverges, so the
        # limit is p = 0 — not the "cannot tell" that a zero denominator
        # might suggest.
        return 0.0

    correction = (1.0 / folds) + (test_size / train_size if train_size else 0.0)
    statistic = mean / float(np.sqrt(correction * variance))
    return float(2.0 * stats.t.sf(abs(statistic), df=folds - 1))


def compare_to_leader(
    scores: dict[str, np.ndarray[Any, Any]],
    *,
    greater_is_better: bool,
    train_size: int,
    test_size: int,
    alpha: float = 0.05,
) -> tuple[Comparison, ...]:
    """Compare every model against the best-scoring one.

    Parameters
    ----------
    scores
        Per-fold scores for each model, keyed by model name.
    greater_is_better
        Direction of the metric being compared.
    train_size, test_size
        Fold sizes, for the variance correction.
    alpha
        Family-wise significance level.

    Returns
    -------
    tuple of Comparison
        One entry per model, ordered best first. The leader compares against
        itself and is never marked significant.
    """
    import numpy as np

    if not scores:
        return ()

    means = {name: float(np.mean(values)) for name, values in scores.items()}
    leader = (max if greater_is_better else min)(means, key=lambda name: means[name])

    raw: dict[str, tuple[float, float]] = {}
    for name, values in scores.items():
        if name == leader:
            continue
        p_value = corrected_paired_t_test(
            scores[leader], values, train_size=train_size, test_size=test_size
        )
        raw[name] = (means[name] - means[leader], p_value)

    adjusted = _holm({name: item[1] for name, item in raw.items()})

    comparisons = [
        Comparison(
            model=leader,
            leader=leader,
            difference=0.0,
            p_value=1.0,
            adjusted_p=1.0,
            significant=False,
        )
    ]
    for name, (difference, p_value) in raw.items():
        comparisons.append(
            Comparison(
                model=name,
                leader=leader,
                difference=difference,
                p_value=p_value,
                adjusted_p=adjusted[name],
                significant=adjusted[name] < alpha,
            )
        )

    comparisons.sort(key=lambda item: means[item.model], reverse=greater_is_better)
    return tuple(comparisons)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    """Adjust p-values with the Holm–Bonferroni step-down procedure.

    Examples
    --------
    >>> adjusted = _holm({"a": 0.01, "b": 0.04})
    >>> adjusted["a"] < adjusted["b"]
    True
    """
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
