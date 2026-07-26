"""Tests for recipes: the model, applying them, and writing them out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from michi.core.errors import RecipeError
from michi.core.io import load_table
from michi.inspection import profile_table
from michi.recipes import (
    Recipe,
    RecipeStep,
    apply_recipe,
    command_for,
    dumps_recipe,
    load_recipe,
    questions_for,
    recipe_from_flags,
    write_recipe,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "keep": [1, 2, 3, 4, 5, 5],
            "drop_me": ["a", "b", "c", "d", "e", "e"],
            "gappy": [1.0, None, 3.0, None, 5.0, 5.0],
            "text_number": ["1,000", "2,000", "3,000", "4,000", "5,000", "5,000"],
            "category": ["x", "y", "x", "y", "x", "x"],
        }
    )


# --- the model -------------------------------------------------------------


def test_unknown_operation_is_rejected() -> None:
    """A typo'd operation names the ones that exist."""
    with pytest.raises(RecipeError, match="known operations"):
        RecipeStep("normalise", {"columns": ["a"]})


def test_unexpected_parameter_is_rejected() -> None:
    """An operation refuses parameters it does not understand."""
    with pytest.raises(RecipeError, match="does not take"):
        RecipeStep("drop", {"columns": ["a"], "strategy": "median"})


def test_fitted_steps_are_identified() -> None:
    """michi knows which steps learn from data, so it can warn about them."""
    assert RecipeStep("impute", {"columns": ["a"]}).is_fitted is True
    assert RecipeStep("drop", {"columns": ["a"]}).is_fitted is False


def test_recipe_round_trips_through_a_dict() -> None:
    """Serialising and rebuilding a recipe preserves its steps."""
    recipe = Recipe(
        steps=(
            RecipeStep("drop", {"columns": ["a"]}, why="constant"),
            RecipeStep("impute", {"columns": ["b"], "strategy": "median"}),
        )
    )
    rebuilt = Recipe.from_dict(recipe.to_dict())
    assert [step.op for step in rebuilt.steps] == ["drop", "impute"]
    assert rebuilt.steps[0].why == "constant"


def test_recipe_without_steps_is_rejected() -> None:
    """A file with no steps list is not a recipe."""
    with pytest.raises(RecipeError, match="steps"):
        Recipe.from_dict({"schema_version": "1.0"})


# --- applying --------------------------------------------------------------


def test_apply_never_modifies_the_input() -> None:
    """The caller's frame is untouched; a new one comes back."""
    frame = _frame()
    before = frame.copy()
    recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["drop_me"]}),))
    apply_recipe(recipe, frame)
    pd.testing.assert_frame_equal(frame, before)


def test_drop_removes_columns() -> None:
    """A drop step removes exactly the columns it names."""
    recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["drop_me"]}),))
    result = apply_recipe(recipe, _frame())
    assert "drop_me" not in result.frame.columns
    assert result.columns_after == result.columns_before - 1


def test_dedupe_removes_repeated_rows() -> None:
    """A dedupe step removes exact duplicates."""
    recipe = Recipe(steps=(RecipeStep("dedupe", {}),))
    result = apply_recipe(recipe, _frame())
    assert result.rows_after == result.rows_before - 1


def test_cast_parses_numbers_behind_separators() -> None:
    """Casting to numeric strips thousands separators."""
    recipe = Recipe(
        steps=(RecipeStep("cast", {"columns": ["text_number"], "to": "numeric"}),)
    )
    result = apply_recipe(recipe, _frame())
    assert result.frame["text_number"].iloc[0] == 1000


def test_impute_fills_missing_values() -> None:
    """Imputation leaves no missing values in the named column."""
    recipe = Recipe(
        steps=(RecipeStep("impute", {"columns": ["gappy"], "strategy": "median"}),)
    )
    result = apply_recipe(recipe, _frame())
    assert result.frame["gappy"].isna().sum() == 0


def test_impute_can_drop_rows_instead() -> None:
    """Dropping affected rows is one of the options, not just filling."""
    recipe = Recipe(
        steps=(RecipeStep("impute", {"columns": ["gappy"], "strategy": "drop_rows"}),)
    )
    result = apply_recipe(recipe, _frame())
    assert result.rows_after == 4


def test_clip_bounds_extreme_values() -> None:
    """Clipping brings values inside the requested quantile range."""
    frame = pd.DataFrame({"value": [*range(1, 60), 1000]})
    recipe = Recipe(
        steps=(
            RecipeStep(
                "clip",
                {"columns": ["value"], "lower_quantile": 0.05, "upper_quantile": 0.95},
            ),
        )
    )
    result = apply_recipe(recipe, frame)
    assert result.frame["value"].max() < 1000


def test_encode_expands_categories() -> None:
    """One-hot encoding produces one column per category."""
    recipe = Recipe(
        steps=(RecipeStep("encode", {"columns": ["category"], "method": "onehot"}),)
    )
    result = apply_recipe(recipe, _frame())
    assert "category_x" in result.frame.columns


def test_scale_centres_a_column() -> None:
    """Standardising leaves a column with approximately zero mean."""
    recipe = Recipe(
        steps=(RecipeStep("scale", {"columns": ["keep"], "method": "standard"}),)
    )
    result = apply_recipe(recipe, _frame())
    assert abs(float(result.frame["keep"].mean())) < 1e-9


def test_steps_apply_in_order() -> None:
    """A column dropped by step one is gone by step two."""
    recipe = Recipe(
        steps=(
            RecipeStep("drop", {"columns": ["gappy"]}),
            RecipeStep("dedupe", {}),
        )
    )
    result = apply_recipe(recipe, _frame())
    assert "gappy" not in result.frame.columns


# --- the schema contract ---------------------------------------------------


def test_missing_schema_columns_fail_loudly(messy_csv: Path) -> None:
    """Applying a recipe to data it does not fit is an error, not a surprise."""
    profile = profile_table(load_table(messy_csv))
    recipe = recipe_from_flags(profile, drop=["notes"])
    with pytest.raises(RecipeError, match="missing"):
        apply_recipe(recipe, pd.DataFrame({"unrelated": [1, 2, 3]}))


def test_non_strict_mode_skips_absent_columns(messy_csv: Path) -> None:
    """`--no-strict` degrades to what still makes sense, and says so."""
    profile = profile_table(load_table(messy_csv))
    recipe = recipe_from_flags(profile, drop=["notes"])
    result = apply_recipe(recipe, pd.DataFrame({"unrelated": [1, 2, 3]}), strict=False)
    assert result.notes


def test_fitted_steps_produce_a_leakage_note() -> None:
    """michi says when it fitted something on every row it was given."""
    recipe = Recipe(
        steps=(RecipeStep("impute", {"columns": ["gappy"], "strategy": "median"}),)
    )
    result = apply_recipe(recipe, _frame())
    assert result.fitted_on_everything


def test_deterministic_recipes_carry_no_leakage_note() -> None:
    """Nothing is warned about when nothing was fitted."""
    recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["drop_me"]}),))
    assert apply_recipe(recipe, _frame()).fitted_on_everything is False


# --- serialisation ---------------------------------------------------------


def test_emitted_yaml_carries_comments() -> None:
    """Recipes are meant to be read, so michi writes comments into them."""
    recipe = Recipe(
        steps=(RecipeStep("drop", {"columns": ["id"]}, why="looked like a key"),)
    )
    text = dumps_recipe(recipe)
    assert "# looked like a key" in text
    assert "michi recipe" in text


def test_emitted_yaml_warns_about_fitted_steps() -> None:
    """The leakage caveat travels with the file, not just the terminal."""
    recipe = Recipe(
        steps=(RecipeStep("impute", {"columns": ["a"], "strategy": "median"}),)
    )
    assert "cannot leak" in dumps_recipe(recipe)


def test_written_recipe_reloads(tmp_path: Path) -> None:
    """A recipe michi writes is a recipe michi can read."""
    recipe = Recipe(
        steps=(
            RecipeStep("drop", {"columns": ["a", "b"]}),
            RecipeStep("impute", {"columns": ["c"], "strategy": "mean"}),
        ),
        target="label",
    )
    path = tmp_path / "recipe.yaml"
    write_recipe(recipe, path)
    reloaded = load_recipe(path)
    assert [step.op for step in reloaded.steps] == ["drop", "impute"]
    assert reloaded.target == "label"
    assert reloaded.steps[0].columns == ("a", "b")


def test_hand_written_recipe_works(tmp_path: Path) -> None:
    """Editing a recipe by hand is a first-class path, not a hack."""
    path = tmp_path / "hand.yaml"
    path.write_text(
        "steps:\n  - op: drop\n    columns: [drop_me]\n  - op: dedupe\n",
        encoding="utf-8",
    )
    result = apply_recipe(load_recipe(path), _frame())
    assert "drop_me" not in result.frame.columns
    assert result.rows_after < result.rows_before


def test_unparseable_recipe_names_the_file(tmp_path: Path) -> None:
    """A YAML syntax error points at the file the user typed into."""
    path = tmp_path / "broken.yaml"
    path.write_text("steps: [ unclosed", encoding="utf-8")
    with pytest.raises(RecipeError, match=r"broken\.yaml"):
        load_recipe(path)


def test_missing_recipe_is_reported(tmp_path: Path) -> None:
    """A missing recipe file fails clearly."""
    with pytest.raises(RecipeError, match="no such recipe"):
        load_recipe(tmp_path / "absent.yaml")


def test_unicode_column_names_survive_a_round_trip(tmp_path: Path) -> None:
    """Non-ASCII column names are written and read back intact."""
    recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["名前"]}),))
    path = tmp_path / "unicode.yaml"
    write_recipe(recipe, path)
    assert load_recipe(path).steps[0].columns == ("名前",)


# --- authoring -------------------------------------------------------------


def test_questions_group_findings_rather_than_columns(messy_csv: Path) -> None:
    """Forty sparse columns become one question, not forty."""
    profile = profile_table(load_table(messy_csv), target="purchased")
    questions = questions_for(profile)
    assert questions
    assert len(questions) < len(profile.findings)


def test_every_question_offers_leaving_things_alone(messy_csv: Path) -> None:
    """Doing nothing is always an option, and always the default."""
    profile = profile_table(load_table(messy_csv), target="purchased")
    for question in questions_for(profile):
        assert any(choice.key == "keep" for choice in question.choices)
        assert question.choices[-1].key == "keep"


def test_flags_produce_ordered_steps(messy_csv: Path) -> None:
    """Steps are ordered so that each one can assume the previous ran."""
    profile = profile_table(load_table(messy_csv))
    recipe = recipe_from_flags(
        profile,
        scale=[("age", "standard")],
        drop=["notes"],
        impute=[("salary", "median")],
        cast=[("amount_text", "numeric")],
    )
    assert [step.op for step in recipe.steps] == ["drop", "cast", "impute", "scale"]


def test_recipe_records_the_schema_it_was_written_against(messy_csv: Path) -> None:
    """The data contract travels with the recipe."""
    profile = profile_table(load_table(messy_csv))
    recipe = recipe_from_flags(profile, drop=["notes"])
    assert recipe.source.sha256
    assert "age" in recipe.source.columns


def test_command_for_reproduces_a_recipe() -> None:
    """A session always converts into something scriptable."""
    recipe = Recipe(
        steps=(
            RecipeStep("drop", {"columns": ["a", "b"]}),
            RecipeStep("impute", {"columns": ["c"], "strategy": "median"}),
        ),
        target="label",
    )
    command = command_for(recipe, "data.csv")
    assert "--drop a,b" in command
    assert "--impute c=median" in command
    assert "--target label" in command
