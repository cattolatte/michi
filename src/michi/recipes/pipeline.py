"""Turning a recipe's fitted steps into an sklearn transformer.

Design Principles
-----------------
- **One meaning, three consumers.** ``apply`` executes a recipe with pandas,
  ``export`` writes it as source code, and this module hands it to
  cross-validation. All three must agree about what the recipe says.
- **Deterministic and fitted steps travel separately.** Dropping, casting,
  and the feature engineering that reads only one row can happen once, up
  front, with no risk. Imputing, encoding, scaling, and quantile binning must
  be fitted per fold, so they become a transformer rather than a mutation.
- michi never silently substitutes its own preparation for a recipe the user
  wrote: when a recipe supplies fitted steps, they are what runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from michi.recipes.model import Recipe, RecipeStep

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["apply_deterministic", "build_transformer", "transformer_specs"]


def apply_deterministic(recipe: Recipe, frame: pd.DataFrame) -> pd.DataFrame:
    """Run only the steps that cannot leak, returning a new frame.

    Dropping a column, deduplicating rows, casting a type, and clipping to
    fixed quantiles depend on nothing learned from other rows' labels, so they
    are safe to apply before splitting.
    """
    from michi.recipes.apply import apply_recipe

    deterministic = Recipe(
        steps=recipe.deterministic_steps,
        target=recipe.target,
    )
    return apply_recipe(deterministic, frame, strict=False).frame


def transformer_specs(
    recipe: Recipe, frame: pd.DataFrame
) -> list[tuple[str, Any, list[str]]]:
    """Return one ``(name, transformer, columns)`` triple per fitted step.

    Callers compose these themselves, because a recipe speaks only about the
    columns it names — and something still has to prepare the rest.
    """
    present = {str(name) for name in frame.columns}
    specs: list[tuple[str, Any, list[str]]] = []

    for index, step in enumerate(recipe.fitted_steps):
        columns = [name for name in step.columns if name in present]
        if not columns:
            continue
        transformer = _transformer_for(step)
        if transformer is None:
            continue
        specs.append((f"{step.op}_{index}", transformer, columns))
    return specs


def build_transformer(recipe: Recipe, frame: pd.DataFrame) -> Any | None:
    """Build a column transformer for a recipe's fitted steps.

    Parameters
    ----------
    recipe
        The recipe whose imputation, encoding, and scaling should be applied.
    frame
        The feature frame, used to check which named columns still exist after
        the deterministic steps ran.

    Returns
    -------
    Any or None
        An sklearn ``ColumnTransformer``, or ``None`` when the recipe has no
        fitted steps and the caller should use its own preparation.
    """
    from sklearn.compose import ColumnTransformer

    transformers = transformer_specs(recipe, frame)
    if not transformers:
        return None

    return ColumnTransformer(
        transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def _transformer_for(step: RecipeStep) -> Any | None:
    """Map one fitted step onto the sklearn transformer that performs it."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        RobustScaler,
        StandardScaler,
    )

    if step.op == "impute":
        strategy = str(step.params.get("strategy", "median"))
        if strategy == "drop_rows":
            # Removing rows is not a column transformation; it belongs to the
            # deterministic pass and is skipped here rather than approximated.
            return None
        if strategy == "constant":
            return SimpleImputer(
                strategy="constant", fill_value=step.params.get("value", 0)
            )
        return SimpleImputer(strategy=strategy)

    if step.op == "encode":
        method = str(step.params.get("method", "onehot"))
        if method == "ordinal":
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if step.op == "scale":
        method = str(step.params.get("method", "standard"))
        return {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }.get(method, StandardScaler())

    if step.op == "bin":
        from sklearn.preprocessing import KBinsDiscretizer

        # `ordinal` keeps one column per input, so the recipe's column names
        # survive into the fitted frame and the deterministic pass that ran
        # before it stays readable in the output.
        return KBinsDiscretizer(
            n_bins=int(step.params.get("bins", 5)),
            encode="ordinal",
            strategy=str(step.params.get("strategy", "quantile")),
            subsample=None,
        )

    return None
