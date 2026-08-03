"""The fold-time pipeline: what gets fitted inside the fold, and what does not.

This module decides which recipe steps are allowed to see the whole dataset
and which must be refitted on every training fold. Getting that wrong does not
raise — it produces a score that is too good, which is the failure mode nobody
notices. It was the least-covered leakage-critical module in the package.
"""

from __future__ import annotations

import pandas as pd
import pytest

from michi.recipes.model import Recipe, RecipeStep
from michi.recipes.pipeline import (
    apply_deterministic,
    build_transformer,
    transformer_specs,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small frame with a numeric hole, a category, and a text column."""
    return pd.DataFrame(
        {
            "age": [30.0, None, 51.0, 44.0, 29.0, 38.0],
            "region": ["north", "south", "north", "west", "south", "west"],
            "note": ["a quick note", "another note", "third", "x", "y", "z"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


# --- the deterministic / fitted boundary ------------------------------------


def test_deterministic_pass_ignores_the_steps_that_learn(frame: pd.DataFrame) -> None:
    """A fitted step must not run before the split, or it has seen the test set."""
    recipe = Recipe(
        steps=(
            RecipeStep("drop", {"columns": ["note"]}),
            RecipeStep("impute", {"columns": ["age"], "strategy": "median"}),
        )
    )
    result = apply_deterministic(recipe, frame)

    assert "note" not in result.columns, "the drop should have run"
    assert result["age"].isna().any(), "the imputation must have been deferred"


def test_deterministic_pass_leaves_the_input_frame_alone(frame: pd.DataFrame) -> None:
    """Callers reuse the frame across folds; mutating it corrupts later folds."""
    recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["note"]}),))
    apply_deterministic(recipe, frame)

    assert "note" in frame.columns


def test_every_fitted_step_reaches_the_transformer(frame: pd.DataFrame) -> None:
    """A fitted step silently dropped here is a step that never runs at all.

    `is_fitted` and `_transformer_for` are two lists that must agree. When they
    disagree the step vanishes: no error, no transformation, and a recipe that
    claims to scale a column it never scaled.
    """
    steps = (
        RecipeStep("impute", {"columns": ["age"], "strategy": "median"}),
        RecipeStep("encode", {"columns": ["region"], "method": "onehot"}),
        RecipeStep("scale", {"columns": ["score"], "method": "standard"}),
        RecipeStep("bin", {"columns": ["score"], "bins": 3}),
        RecipeStep("tfidf", {"columns": ["note"], "max_features": 8}),
        RecipeStep(
            "target-encode", {"columns": ["region"], "target": "y", "smoothing": 10.0}
        ),
    )
    for step in steps:
        assert step.is_fitted, f"{step.op} should be fitted"
        specs = transformer_specs(Recipe(steps=(step,)), frame)
        assert specs, f"{step.op} claims to be fitted but builds no transformer"


# --- selection and shape ----------------------------------------------------


def test_a_column_the_recipe_names_but_the_frame_lost_is_skipped(
    frame: pd.DataFrame,
) -> None:
    """`clean` may drop a column a later step names; that is not a crash."""
    recipe = Recipe(
        steps=(RecipeStep("scale", {"columns": ["missing_entirely"]}),),
    )
    assert transformer_specs(recipe, frame) == []
    assert build_transformer(recipe, frame) is None


def test_a_recipe_with_no_fitted_steps_builds_nothing(frame: pd.DataFrame) -> None:
    """None tells the caller to use its own preparation instead of an empty one."""
    recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["note"]}),))
    assert build_transformer(recipe, frame) is None


def test_tfidf_gets_one_spec_per_column_not_one_per_step(
    frame: pd.DataFrame,
) -> None:
    """A vectorizer takes a Series; a list selector hands it a DataFrame.

    ColumnTransformer only passes a Series when the selector is a bare string,
    so two text columns need two specs. One spec with a two-name list raises
    deep inside sklearn with a message about `lower` not existing on a frame.
    """
    frame = frame.assign(other="more text here")
    recipe = Recipe(steps=(RecipeStep("tfidf", {"columns": ["note", "other"]}),))
    specs = transformer_specs(recipe, frame)

    assert len(specs) == 2
    assert all(isinstance(columns, str) for _, _, columns in specs)


def test_spec_names_stay_unique_across_repeated_operations(
    frame: pd.DataFrame,
) -> None:
    """ColumnTransformer rejects duplicate names, and recipes repeat ops freely."""
    recipe = Recipe(
        steps=(
            RecipeStep("scale", {"columns": ["age"], "method": "standard"}),
            RecipeStep("scale", {"columns": ["score"], "method": "minmax"}),
        )
    )
    names = [name for name, _, _ in transformer_specs(recipe, frame)]
    assert len(names) == len(set(names))


def test_drop_rows_imputation_builds_no_column_transformer(
    frame: pd.DataFrame,
) -> None:
    """Removing rows is not a column transformation and must not be faked as one."""
    recipe = Recipe(
        steps=(RecipeStep("impute", {"columns": ["age"], "strategy": "drop_rows"}),)
    )
    assert transformer_specs(recipe, frame) == []


# --- it actually has to work ------------------------------------------------


def test_the_built_transformer_fits_and_leaves_no_holes(frame: pd.DataFrame) -> None:
    """The point of all of this is a numeric matrix a model can consume."""
    recipe = Recipe(
        steps=(
            RecipeStep("impute", {"columns": ["age"], "strategy": "median"}),
            RecipeStep("encode", {"columns": ["region"], "method": "onehot"}),
            RecipeStep("scale", {"columns": ["score"], "method": "robust"}),
        )
    )
    transformer = build_transformer(recipe, frame)
    assert transformer is not None

    out = transformer.fit_transform(frame.drop(columns=["note"]))
    assert out.shape[0] == len(frame)
    assert not pd.DataFrame(out).isna().to_numpy().any()


def test_unnamed_columns_survive_rather_than_being_discarded(
    frame: pd.DataFrame,
) -> None:
    """`remainder="passthrough"`: a recipe speaks only about what it names."""
    recipe = Recipe(steps=(RecipeStep("scale", {"columns": ["score"]}),))
    transformer = build_transformer(recipe, frame)
    assert transformer is not None

    numeric = frame[["age", "score"]].fillna(0.0)
    assert transformer.fit_transform(numeric).shape[1] == 2


@pytest.mark.parametrize("method", ["standard", "minmax", "robust", "nonsense"])
def test_every_scaler_method_yields_a_working_scaler(
    frame: pd.DataFrame, method: str
) -> None:
    """Including an unknown one, which falls back rather than returning None."""
    recipe = Recipe(
        steps=(RecipeStep("scale", {"columns": ["score"], "method": method}),)
    )
    transformer = build_transformer(recipe, frame)
    assert transformer is not None
    assert transformer.fit_transform(frame[["score"]]).shape == (len(frame), 1)


@pytest.mark.parametrize("method", ["onehot", "ordinal"])
def test_both_encoders_tolerate_a_category_absent_from_training(
    frame: pd.DataFrame, method: str
) -> None:
    """A fold-time encoder meets unseen categories constantly; it must not raise."""
    recipe = Recipe(
        steps=(RecipeStep("encode", {"columns": ["region"], "method": method}),)
    )
    transformer = build_transformer(recipe, frame)
    assert transformer is not None

    transformer.fit(frame[["region"]])
    unseen = pd.DataFrame({"region": ["atlantis"]})
    assert transformer.transform(unseen).shape[0] == 1
