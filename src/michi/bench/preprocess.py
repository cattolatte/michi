"""Minimal preparation so that raw columns can reach an estimator.

Design Principles
-----------------
- **Stated, never hidden.** Every model needs numeric input, so ``bench`` must
  do *something* with missing values and categories. What it does is named in
  the terminal output and recorded in every manifest, and each step is
  overridable by flag. An assumption a user can read is not a decision michi
  made for them.
- **Fitted inside each fold, always.** Imputers and encoders learn from the
  training fold only. Fitting them on the whole dataset first is the most
  common way a benchmark quietly leaks and reports scores nobody can
  reproduce.
- **Deliberately minimal.** This is the least transformation that lets a model
  train — not feature engineering. Real cleaning decisions belong in a recipe
  the user authors (``michi clean``, v0.4), which ``bench`` will accept.
- **Scaling follows the model.** Standardisation is applied only for models
  that are sensitive to feature scale, because it is mechanics for them and
  noise for tree ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

__all__ = [
    "PreparationPolicy",
    "build_pipeline",
    "column_specs",
    "describe_policy",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_MAX_ONEHOT_CARDINALITY: Final = 50


@dataclass(frozen=True, slots=True)
class PreparationPolicy:
    """How raw columns are made model-ready, as an explicit, recorded choice.

    Examples
    --------
    >>> PreparationPolicy().numeric_impute
    'median'
    """

    numeric_impute: str = "median"
    categorical_impute: str = "most_frequent"
    encode: str = "onehot"
    scale: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in a run manifest."""
        return {
            "numeric_impute": self.numeric_impute,
            "categorical_impute": self.categorical_impute,
            "encode": self.encode,
            "scale": self.scale,
        }


def describe_policy(policy: PreparationPolicy, *, scaled: bool) -> str:
    """One line naming exactly what will be done to the columns."""
    parts = [
        f"numeric: impute {policy.numeric_impute}",
        f"categorical: impute {policy.categorical_impute} + {policy.encode}",
    ]
    if policy.scale and scaled:
        parts.append("standardise (scale-sensitive models only)")
    return " · ".join(parts) + " — fitted inside each fold"


def column_specs(
    frame: pd.DataFrame,
    policy: PreparationPolicy,
    *,
    needs_scaling: bool,
    skip: set[str] | None = None,
) -> list[tuple[str, Any, list[str]]]:
    """Default preparation for the columns nobody else has claimed.

    A recipe speaks only about the columns it names. The rest still have to
    reach the estimator as numbers, so michi prepares them the documented way
    — and says so, rather than passing strings through to a model that will
    fail on them.
    """
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    claimed = skip or set()
    numeric: list[str] = []
    categorical: list[str] = []
    for name in frame.columns:
        if str(name) in claimed:
            continue
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric.append(str(name))
        else:
            categorical.append(str(name))

    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy=policy.numeric_impute))
    ]
    if policy.scale and needs_scaling:
        numeric_steps.append(("scale", StandardScaler()))

    if policy.encode == "ordinal":
        encoder: Any = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=_MAX_ONEHOT_CARDINALITY,
        )

    categorical_steps: list[tuple[str, Any]] = [
        (
            "impute",
            SimpleImputer(strategy=policy.categorical_impute, fill_value="missing"),
        ),
        ("encode", encoder),
    ]

    specs: list[tuple[str, Any, list[str]]] = []
    if numeric:
        specs.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        specs.append(("categorical", Pipeline(categorical_steps), categorical))
    return specs


def build_pipeline(
    frame: pd.DataFrame,
    estimator: Any,
    policy: PreparationPolicy,
    *,
    needs_scaling: bool,
) -> Any:
    """Wrap an estimator in the column preparation it needs.

    Parameters
    ----------
    frame
        The feature frame, used only to decide which columns are numeric and
        which are categorical.
    estimator
        The model to place at the end of the pipeline.
    policy
        The preparation choices to apply.
    needs_scaling
        Whether this estimator is sensitive to feature scale.

    Returns
    -------
    Any
        An sklearn ``Pipeline`` whose every fitted step learns from training
        data only.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    specs = column_specs(frame, policy, needs_scaling=needs_scaling)
    preparation = ColumnTransformer(specs, remainder="drop")
    return Pipeline([("prepare", preparation), ("model", estimator)])
