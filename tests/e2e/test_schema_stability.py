"""The 1.0 schema freeze, enforced.

ADR-0002 promises that an artifact michi wrote at 1.0 stays readable. A
promise nobody tests is a hope, so these tests read committed 1.0 fixtures on
every CI run.

If a change here fails, the change is wrong — not the fixture. Editing a
fixture to make a test pass would silently break every artifact a user already
has on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from michi.core.artifacts import PROFILE_SCHEMA_VERSION, DatasetProfile
from michi.core.manifest import MANIFEST_SCHEMA_VERSION, RunManifest
from michi.recipes import RECIPE_SCHEMA_VERSION, load_recipe

FIXTURES = Path(__file__).parent.parent / "fixtures" / "schemas"


# --- the profile artifact --------------------------------------------------


def test_a_1_0_profile_still_loads() -> None:
    """A profile written at 1.0 is readable by this version."""
    payload = json.loads((FIXTURES / "profile-1.0.json").read_text(encoding="utf-8"))
    profile = DatasetProfile.from_dict(payload)

    assert profile.schema_version == "1.0"
    assert profile.n_rows == 120
    assert profile.target == "region"
    assert profile.column("age").stats["mean"] == pytest.approx(41.6)
    assert profile.column("region").top_values[0] == ("north", 60)
    assert profile.findings[0].kind == "missing"
    assert profile.source.sha256 == "a" * 64


def test_the_profile_schema_version_has_not_moved() -> None:
    """Bumping this constant is a breaking change, not a routine edit."""
    assert PROFILE_SCHEMA_VERSION == "1.0"


def test_a_1_0_profile_round_trips_unchanged() -> None:
    """Reading and rewriting a 1.0 profile preserves every field."""
    payload = json.loads((FIXTURES / "profile-1.0.json").read_text(encoding="utf-8"))
    assert DatasetProfile.from_dict(payload).to_dict() == payload


# --- the run manifest ------------------------------------------------------


def test_a_1_0_manifest_still_loads() -> None:
    """A manifest written at 1.0 is readable by this version."""
    payload = json.loads((FIXTURES / "manifest-1.0.json").read_text(encoding="utf-8"))
    manifest = RunManifest.from_dict(payload)

    assert manifest.schema_version == "1.0"
    assert manifest.kind == "eval"
    assert manifest.task == "classification"
    assert manifest.primary.name == "balanced_accuracy"
    assert manifest.primary.ci_low == pytest.approx(0.6098)
    assert manifest.metric("log_loss").greater_is_better is False
    assert manifest.baselines["most_frequent"][0].value == pytest.approx(0.5)
    assert manifest.environment.packages["scikit-learn"] == "1.7.0"


def test_the_manifest_schema_version_has_not_moved() -> None:
    """Bumping this constant is a breaking change, not a routine edit."""
    assert MANIFEST_SCHEMA_VERSION == "1.0"


def test_a_1_0_manifest_round_trips_unchanged() -> None:
    """Reading and rewriting a 1.0 manifest preserves every field."""
    payload = json.loads((FIXTURES / "manifest-1.0.json").read_text(encoding="utf-8"))
    assert RunManifest.from_dict(payload).to_dict() == payload


# --- the recipe ------------------------------------------------------------


def test_a_1_0_recipe_still_loads() -> None:
    """A recipe written at 1.0 is readable, with every operation intact."""
    recipe = load_recipe(FIXTURES / "recipe-1.0.yaml")

    assert recipe.schema_version == "1.0"
    assert recipe.target == "purchased"
    assert [step.op for step in recipe.steps] == [
        "drop",
        "cast",
        "impute",
        "clip",
        "encode",
        "scale",
        "dedupe",
    ]
    assert recipe.steps[0].why == "column was entirely missing"
    assert recipe.steps[2].params["strategy"] == "median"
    assert recipe.source.columns["age"] == "numeric"


def test_the_recipe_schema_version_has_not_moved() -> None:
    """Bumping this constant is a breaking change, not a routine edit."""
    assert RECIPE_SCHEMA_VERSION == "1.0"


def test_every_1_0_operation_still_applies() -> None:
    """Each frozen operation still executes, not merely parses."""
    import pandas as pd

    from michi.recipes import apply_recipe

    recipe = load_recipe(FIXTURES / "recipe-1.0.yaml")
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 40],
            "region": ["north", "south", "north", "north"],
            "notes": [None, None, None, None],
            "amount": ["1,000", "2,000", "3,000", "3,000"],
            "salary": [100.0, None, 300.0, 300.0],
            "fare": [1.0, 2.0, 900.0, 900.0],
        }
    )
    result = apply_recipe(recipe, frame, strict=False)
    assert "notes" not in result.frame.columns
    assert result.frame["salary"].isna().sum() == 0


def test_a_1_0_recipe_still_compiles_to_code() -> None:
    """Exported code is part of the promise, not only the artifact."""
    from michi.recipes import export_recipe

    code = export_recipe(load_recipe(FIXTURES / "recipe-1.0.yaml"))
    assert "def prepare(" in code
    assert "def build_pipeline(" in code


# --- the plugin contract ---------------------------------------------------


def test_the_plugin_entry_point_groups_are_frozen() -> None:
    """Plugin authors publish against these names; they cannot move."""
    from michi.plugins import ADAPTER_GROUP, MODEL_GROUP

    assert MODEL_GROUP == "michi.models"
    assert ADAPTER_GROUP == "michi.adapters"


# --- the CLI surface -------------------------------------------------------


def test_every_frozen_verb_is_present() -> None:
    """The verb names are public API from 1.0."""
    from michi.cli.app import app

    registered = {
        command.name or (command.callback.__name__ if command.callback else "")
        for command in app.registered_commands
    }
    frozen = {
        "inspect",
        "eval",
        "bench",
        "report",
        "clean",
        "apply",
        "export",
        "sweep",
        "ui",
        "plugins",
        "info",
    }
    assert frozen <= registered
