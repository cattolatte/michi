"""What the model actually leans on, measured by taking it away.

Design Principles
-----------------
- **Measured through the same door everything else uses.** Importance is
  computed by shuffling a column and watching the score fall, so it needs only
  ``predict``. A PyTorch network, an ONNX graph, and a random forest are
  measured identically, and michi needs no per-model introspection it would
  have to keep working forever.
- **It answers "what does this model use", not "what matters".** A column the
  model ignores may still drive the outcome — the model simply found the
  signal somewhere else. That distinction is the difference between a
  diagnostic and a causal claim, and it is stated wherever the numbers appear.
- **Correlated columns split the credit and michi says so.** Two duplicated
  features each look unimportant, because shuffling either leaves the other
  to carry the signal. Reporting a rank without that caveat has misled people
  into deleting a feature that mattered.
- **A random procedure gets error bars.** Each column is shuffled several
  times and the spread is reported, because an importance smaller than its own
  noise is not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from michi.adapters import LoadedModel

__all__ = ["ColumnImportance", "permutation_importance"]

DEFAULT_REPEATS = 5
"""Shuffles per column. Enough for a spread, cheap enough to stay default."""

_MAX_COLUMNS = 60
"""Above this, importance costs more than it tells anyone."""


@dataclass(frozen=True, slots=True)
class ColumnImportance:
    """How much one column's information is worth to this model.

    Attributes
    ----------
    column
        The column that was shuffled.
    drop
        Mean fall in the score when it was. Positive means the model was using
        it; near zero means it was not; negative means the model did slightly
        better without it, which is noise rather than a discovery.
    spread
        Standard deviation of the fall across repeats.
    """

    column: str
    drop: float
    spread: float

    @property
    def is_noise(self) -> bool:
        """Whether the measured drop is smaller than its own variation.

        An importance inside its own error bar is a column this run could not
        distinguish from one the model ignores.

        Examples
        --------
        >>> ColumnImportance("age", drop=0.20, spread=0.01).is_noise
        False
        >>> ColumnImportance("noise", drop=0.002, spread=0.01).is_noise
        True
        """
        return abs(self.drop) <= self.spread

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in a run manifest."""
        return {
            "column": self.column,
            "drop": round(self.drop, 6),
            "spread": round(self.spread, 6),
        }


def permutation_importance(
    model: LoadedModel,
    frame: pd.DataFrame,
    labels: Any,
    *,
    scorer: Any,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
    columns: tuple[str, ...] | None = None,
) -> tuple[ColumnImportance, ...]:
    """Measure each column by shuffling it and watching the score fall.

    Parameters
    ----------
    model
        A loaded model; only ``predict`` is called.
    frame
        The feature frame the model was evaluated on.
    labels
        True labels, aligned with `frame`.
    scorer
        ``scorer(truth, predictions) -> float``, higher being better.
    repeats
        Shuffles per column.
    seed
        Seed for the shuffles, so the result repeats.
    columns
        Columns to measure; defaults to all of them.

    Returns
    -------
    tuple of ColumnImportance
        One entry per column, largest drop first. Empty when the frame is too
        wide to measure without costing more than it tells anyone.
    """
    import numpy as np

    names = list(columns or frame.columns)
    if not names or len(names) > _MAX_COLUMNS or frame.empty:
        return ()

    baseline = float(scorer(labels, model.predict(frame)))
    rng = np.random.default_rng(seed)
    measured: list[ColumnImportance] = []

    for name in names:
        if name not in frame.columns:
            continue
        original = frame[name].to_numpy(copy=True)
        drops: list[float] = []
        working = frame.copy()
        for _ in range(repeats):
            # Shuffling breaks this column's relationship with the label while
            # leaving its marginal distribution intact — so the model still
            # receives values it considers plausible, and the fall measures
            # lost information rather than a rejected input.
            working[name] = rng.permutation(original)
            try:
                score = float(scorer(labels, model.predict(working)))
            except Exception:  # third-party failure boundary
                # A model that rejects a shuffled column tells us nothing
                # about that column; it is skipped rather than scored zero,
                # which would read as "unimportant".
                drops = []
                break
            drops.append(baseline - score)
        if not drops:
            continue
        values = np.asarray(drops, dtype=float)
        measured.append(
            ColumnImportance(
                column=str(name),
                drop=float(values.mean()),
                spread=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            )
        )

    measured.sort(key=lambda item: item.drop, reverse=True)
    return tuple(measured)
