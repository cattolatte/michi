"""Cleaning recipes — the ``michi clean`` / ``apply`` / ``export`` verbs.

A recipe is the artifact that makes cleaning reproducible: an ordered list of
declarative operations plus the schema they were written against. ``clean``
authors one, ``apply`` executes it non-destructively, and ``export`` compiles
it into readable pipeline code the user owns.

Design Principles
-----------------
- The session is the authoring interface; the recipe is the product.
- Cleaning never modifies the input data. Only ``apply`` produces data, and
  always to a new file.
- ``apply`` and ``export`` must mean the same thing by a recipe.
- Fitted steps are marked as such, so the leakage risk is visible rather than
  discovered later.
"""

from __future__ import annotations

from michi.recipes.apply import ApplyResult, apply_recipe
from michi.recipes.author import (
    Choice,
    Opportunity,
    Question,
    command_for,
    feature_opportunities,
    ordered_steps,
    questions_for,
    recipe_from_answers,
    recipe_from_flags,
)
from michi.recipes.export import export_recipe
from michi.recipes.model import (
    RECIPE_SCHEMA_VERSION,
    Recipe,
    RecipeStep,
    SourceSchema,
)
from michi.recipes.pipeline import (
    apply_deterministic,
    build_transformer,
    transformer_specs,
)
from michi.recipes.serialise import dumps_recipe, load_recipe, write_recipe

__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "ApplyResult",
    "Choice",
    "Opportunity",
    "Question",
    "Recipe",
    "RecipeStep",
    "SourceSchema",
    "apply_deterministic",
    "apply_recipe",
    "build_transformer",
    "command_for",
    "dumps_recipe",
    "export_recipe",
    "feature_opportunities",
    "load_recipe",
    "ordered_steps",
    "questions_for",
    "recipe_from_answers",
    "recipe_from_flags",
    "transformer_specs",
    "write_recipe",
]
