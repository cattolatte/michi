"""Executing a recipe against data.

Design Principles
-----------------
- **Never destructive.** ``apply`` returns a new frame; the input file is
  never touched. Writing is the caller's separate, explicit act.
- **Loud about the schema.** A recipe carries the columns it was written
  against. Applying it to data missing those columns fails with a message
  naming them, rather than half-transforming the frame.
- **Honest about leakage.** Fitted steps — imputation, encoding, scaling —
  learn from whatever rows they are given. When ``apply`` fits them on a whole
  file, it says so, and points at ``michi export`` for the modelling path
  where fitting must happen inside the split.
- ``apply`` and ``export`` must produce the same result; a test asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from michi.core.errors import RecipeError
from michi.recipes.model import Recipe, RecipeStep

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["ApplyResult", "apply_recipe"]


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """The outcome of applying a recipe."""

    frame: pd.DataFrame
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fitted_on_everything(self) -> bool:
        """Whether any step learned from every row it was given."""
        return any("fitted on all" in note for note in self.notes)


def apply_recipe(
    recipe: Recipe, frame: pd.DataFrame, *, strict: bool = True
) -> ApplyResult:
    """Execute a recipe's steps against a dataframe.

    Parameters
    ----------
    recipe
        The recipe to execute.
    frame
        Input data. It is never modified; a transformed copy is returned.
    strict
        When true, a column named by the recipe but absent from the data is an
        error. When false, such steps are skipped and reported as notes.

    Returns
    -------
    ApplyResult
        The transformed frame, shape before and after, and any notes worth
        surfacing to the user.

    Raises
    ------
    RecipeError
        If the data does not satisfy the recipe's schema, or a step's
        parameters are invalid.
    """
    working = frame.copy()
    rows_before = int(working.shape[0])
    columns_before = int(working.shape[1])
    notes: list[str] = []

    _check_schema(recipe, working, strict=strict, notes=notes)

    for index, step in enumerate(recipe.steps):
        try:
            working = _apply_step(step, working, strict=strict, notes=notes)
        except RecipeError:
            raise
        except Exception as err:  # third-party failure boundary
            msg = f"step {index + 1} ({step.op}) failed: {err}"
            raise RecipeError(msg) from err

    if recipe.fitted_steps:
        names = ", ".join(sorted({step.op for step in recipe.fitted_steps}))
        notes.append(
            f"{names} fitted on all {rows_before:,} rows of this file — for "
            "modelling, fit inside the train/test split instead "
            "(michi export writes a pipeline that does)"
        )

    return ApplyResult(
        frame=working,
        rows_before=rows_before,
        rows_after=int(working.shape[0]),
        columns_before=columns_before,
        columns_after=int(working.shape[1]),
        notes=tuple(notes),
    )


def _check_schema(
    recipe: Recipe, frame: pd.DataFrame, *, strict: bool, notes: list[str]
) -> None:
    """Verify the data satisfies the contract the recipe was written against."""
    expected = set(recipe.source.columns)
    if not expected:
        return

    present = {str(name) for name in frame.columns}
    missing = sorted(expected - present)
    if missing and strict:
        msg = (
            f"data is missing {len(missing)} column(s) the recipe was written "
            f"against: {', '.join(missing[:8])}. Pass --no-strict to apply the "
            "steps that still make sense."
        )
        raise RecipeError(msg)
    if missing:
        notes.append(
            f"{len(missing)} recipe column(s) absent: {', '.join(missing[:5])}"
        )

    extra = sorted(present - expected)
    if extra:
        notes.append(
            f"{len(extra)} column(s) not in the recipe's schema: {', '.join(extra[:5])}"
        )


def _apply_step(
    step: RecipeStep, frame: pd.DataFrame, *, strict: bool, notes: list[str]
) -> pd.DataFrame:
    """Dispatch one step."""
    handlers = {
        "drop": _apply_drop,
        "dedupe": _apply_dedupe,
        "cast": _apply_cast,
        "impute": _apply_impute,
        "clip": _apply_clip,
        "encode": _apply_encode,
        "scale": _apply_scale,
    }
    present = _resolve_columns(step, frame, strict=strict, notes=notes)
    if step.columns and not present and step.op != "dedupe":
        return frame
    return handlers[step.op](step, frame, present)


def _resolve_columns(
    step: RecipeStep, frame: pd.DataFrame, *, strict: bool, notes: list[str]
) -> list[str]:
    """Narrow a step's columns to those actually present."""
    named = step.columns
    if not named:
        return []
    present = {str(name) for name in frame.columns}
    missing = [name for name in named if name not in present]
    if missing and strict:
        msg = f"step {step.op!r} names column(s) not in the data: {', '.join(missing)}"
        raise RecipeError(msg)
    if missing:
        notes.append(f"{step.op}: skipped absent column(s) {', '.join(missing)}")
    return [name for name in named if name in present]


def _apply_drop(
    step: RecipeStep, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    return frame.drop(columns=columns)


def _apply_dedupe(
    step: RecipeStep, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    subset = columns or None
    try:
        return frame.drop_duplicates(subset=subset).reset_index(drop=True)
    except TypeError:
        # Unhashable cell values (lists, dicts) defeat pandas' hashing; a
        # string view is a stable fallback for identifying duplicates.
        mask = frame.astype(str).duplicated(subset=subset)
        return frame.loc[~mask].reset_index(drop=True)


def _apply_cast(
    step: RecipeStep, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    import pandas as pd

    target_type = str(step.params.get("to", "numeric"))
    result = frame.copy()
    for name in columns:
        if target_type == "numeric":
            cleaned = (
                result[name].astype(str).str.replace(r"[,\s$€£¥%_]", "", regex=True)
            )
            result[name] = pd.to_numeric(cleaned, errors="coerce")
        elif target_type == "datetime":
            result[name] = pd.to_datetime(result[name], errors="coerce", format="mixed")
        elif target_type == "category":
            result[name] = result[name].astype("category")
        elif target_type == "string":
            result[name] = result[name].astype(str)
        else:
            msg = (
                f"cast: unknown target type {target_type!r}; "
                "expected numeric, datetime, category, or string"
            )
            raise RecipeError(msg)
    return result


def _apply_impute(
    step: RecipeStep, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    import pandas as pd

    strategy = str(step.params.get("strategy", "median"))
    result = frame.copy()

    if strategy == "drop_rows":
        return result.dropna(subset=columns).reset_index(drop=True)

    for name in columns:
        series = result[name]
        if strategy == "median":
            fill: Any = pd.to_numeric(series, errors="coerce").median()
        elif strategy == "mean":
            fill = pd.to_numeric(series, errors="coerce").mean()
        elif strategy == "most_frequent":
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if not modes.empty else None
        elif strategy == "constant":
            fill = step.params.get("value", 0)
        else:
            msg = (
                f"impute: unknown strategy {strategy!r}; expected median, "
                "mean, most_frequent, constant, or drop_rows"
            )
            raise RecipeError(msg)
        if fill is not None and fill == fill:
            result[name] = series.fillna(fill)
    return result


def _apply_clip(
    step: RecipeStep, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    import pandas as pd

    lower_q = float(step.params.get("lower_quantile", 0.01))
    upper_q = float(step.params.get("upper_quantile", 0.99))
    if not 0.0 <= lower_q < upper_q <= 1.0:
        msg = (
            f"clip: quantiles must satisfy 0 <= lower < upper <= 1 "
            f"(got {lower_q} and {upper_q})"
        )
        raise RecipeError(msg)

    result = frame.copy()
    for name in columns:
        numeric = pd.to_numeric(result[name], errors="coerce")
        lower = numeric.quantile(lower_q)
        upper = numeric.quantile(upper_q)
        result[name] = numeric.clip(lower=lower, upper=upper)
    return result


def _apply_encode(
    step: RecipeStep, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    import pandas as pd

    method = str(step.params.get("method", "onehot"))
    result = frame.copy()

    if method == "onehot":
        return pd.get_dummies(result, columns=columns, dummy_na=False)
    if method == "ordinal":
        for name in columns:
            codes = result[name].astype("category").cat.codes
            result[name] = codes.replace(-1, None)
        return result
    msg = f"encode: unknown method {method!r}; expected onehot or ordinal"
    raise RecipeError(msg)


def _apply_scale(
    step: RecipeStep, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    import pandas as pd

    method = str(step.params.get("method", "standard"))
    result = frame.copy()
    for name in columns:
        numeric = pd.to_numeric(result[name], errors="coerce")
        if method == "standard":
            spread = numeric.std()
            result[name] = (numeric - numeric.mean()) / (spread if spread else 1.0)
        elif method == "minmax":
            low, high = numeric.min(), numeric.max()
            span = high - low
            result[name] = (numeric - low) / (span if span else 1.0)
        elif method == "robust":
            spread = numeric.quantile(0.75) - numeric.quantile(0.25)
            result[name] = (numeric - numeric.median()) / (spread if spread else 1.0)
        else:
            msg = (
                f"scale: unknown method {method!r}; expected standard, "
                "minmax, or robust"
            )
            raise RecipeError(msg)
    return result
