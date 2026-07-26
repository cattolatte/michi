"""End-to-end tests for exported pipeline code.

The claim ``michi export`` makes is that the generated file means the same
thing as the recipe. These tests execute the generated code and check it —
because a code generator whose output is merely plausible is worse than no
code generator at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from michi.recipes import Recipe, RecipeStep, apply_recipe, export_recipe


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "keep": [1, 2, 3, 4, 5, 5, 7, 8],
            "drop_me": ["a", "b", "c", "d", "e", "e", "f", "g"],
            "text_number": [
                "1,000",
                "2,000",
                "3,000",
                "4,000",
                "5,000",
                "5,000",
                "6,000",
                "7,000",
            ],
            "wide": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 900.0],
        }
    )


def _deterministic_recipe() -> Recipe:
    return Recipe(
        steps=(
            RecipeStep("drop", {"columns": ["drop_me"]}, why="not a feature"),
            RecipeStep("dedupe", {}),
            RecipeStep("cast", {"columns": ["text_number"], "to": "numeric"}),
            RecipeStep(
                "clip",
                {
                    "columns": ["wide"],
                    "lower_quantile": 0.05,
                    "upper_quantile": 0.95,
                },
            ),
        ),
        target="keep",
    )


def _write_and_import(code: str, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Write generated code to a module and import it."""
    import importlib.util

    module_path = tmp_path / "generated_pipeline.py"
    module_path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("generated_pipeline", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- equivalence -----------------------------------------------------------


def test_generated_prepare_matches_apply(tmp_path: Path) -> None:
    """The exported code and `michi apply` mean the same thing.

    This is the contract that makes `export` trustworthy: a user who leaves
    michi behind must get exactly the transformation they had.
    """
    recipe = _deterministic_recipe()
    module = _write_and_import(export_recipe(recipe), tmp_path)

    from_export = module.prepare(_frame())
    from_apply = apply_recipe(recipe, _frame()).frame

    pd.testing.assert_frame_equal(
        from_export.reset_index(drop=True),
        from_apply.reset_index(drop=True),
        check_dtype=False,
    )


def test_generated_code_runs_on_new_data(tmp_path: Path) -> None:
    """Exported preparation works on data it has never seen."""
    module = _write_and_import(export_recipe(_deterministic_recipe()), tmp_path)
    fresh = _frame().iloc[:4]
    assert "drop_me" not in module.prepare(fresh).columns


def test_generated_pipeline_fits_and_transforms(tmp_path: Path) -> None:
    """The fitted half of the export is a working sklearn transformer."""
    recipe = Recipe(
        steps=(
            RecipeStep("impute", {"columns": ["wide"], "strategy": "median"}),
            RecipeStep("scale", {"columns": ["keep"], "method": "standard"}),
        )
    )
    module = _write_and_import(export_recipe(recipe), tmp_path)
    transformer = module.build_pipeline()

    frame = _frame()[["keep", "wide"]]
    transformed = transformer.fit_transform(frame)
    assert transformed.shape[0] == frame.shape[0]


def test_export_without_fitted_steps_returns_no_pipeline(tmp_path: Path) -> None:
    """A purely deterministic recipe honestly has no pipeline to build."""
    module = _write_and_import(export_recipe(_deterministic_recipe()), tmp_path)
    assert module.build_pipeline() is None


# --- quality of the generated file -----------------------------------------


def test_generated_code_passes_ruff(tmp_path: Path) -> None:
    """Generated code must satisfy the same linter michi itself does."""
    path = tmp_path / "pipeline.py"
    path.write_text(export_recipe(_deterministic_recipe()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 1 and "No module named" in result.stderr:
        pytest.skip("ruff is not installed in this environment")
    assert result.returncode == 0, result.stdout


def test_generated_code_is_already_formatted(tmp_path: Path) -> None:
    """Generated code is emitted in the form the formatter would produce."""
    path = tmp_path / "pipeline.py"
    path.write_text(export_recipe(_deterministic_recipe()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 1 and "No module named" in result.stderr:
        pytest.skip("ruff is not installed in this environment")
    assert result.returncode == 0, result.stdout


def test_generated_code_never_imports_michi() -> None:
    """A user who outgrows michi leaves with code, not a dependency."""
    code = export_recipe(_deterministic_recipe())
    assert "michi" not in code.replace("michi recipe", "").replace(
        "compiled from a michi", ""
    ).replace("michi is not a runtime dependency", "").replace("michi export", "")


def test_generated_code_explains_the_leakage_split() -> None:
    """The reason for two functions is stated in the file itself."""
    code = export_recipe(
        Recipe(steps=(RecipeStep("impute", {"columns": ["a"], "strategy": "median"}),))
    )
    assert "cross-validation" in code
