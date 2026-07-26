"""Tests for the `michi clean`, `apply`, and `export` commands."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from michi.cli.app import app
from michi.recipes import load_recipe

runner = CliRunner()

# --- authoring -------------------------------------------------------------


def test_clean_writes_a_recipe_from_flags(messy_csv: Path, tmp_path: Path) -> None:
    """Flags alone produce a recipe — no prompting required."""
    destination = tmp_path / "recipe.yaml"
    result = runner.invoke(
        app,
        [
            "clean",
            str(messy_csv),
            "--drop",
            "notes,country",
            "--dedupe",
            "-o",
            str(destination),
        ],
    )
    assert result.exit_code == 0
    recipe = load_recipe(destination)
    assert [step.op for step in recipe.steps] == ["drop", "dedupe"]


def test_clean_never_touches_the_data(messy_csv: Path, tmp_path: Path) -> None:
    """Authoring a recipe leaves the dataset byte-identical."""
    before = messy_csv.read_bytes()
    runner.invoke(
        app,
        ["clean", str(messy_csv), "--drop", "notes", "-o", str(tmp_path / "r.yaml")],
    )
    assert messy_csv.read_bytes() == before


def test_clean_prints_the_reproducing_command(messy_csv: Path, tmp_path: Path) -> None:
    """Every session converts into something scriptable."""
    result = runner.invoke(
        app,
        [
            "clean",
            str(messy_csv),
            "--drop",
            "notes",
            "--impute",
            "salary=median",
            "-o",
            str(tmp_path / "r.yaml"),
        ],
    )
    assert "To reproduce" in result.output
    assert "--impute salary=median" in result.output.replace("\n", " ")


def test_clean_accepts_repeated_pair_options(messy_csv: Path, tmp_path: Path) -> None:
    """Column=value options can be repeated for several columns."""
    destination = tmp_path / "r.yaml"
    result = runner.invoke(
        app,
        [
            "clean",
            str(messy_csv),
            "--impute",
            "salary=median",
            "--cast",
            "amount_text=numeric",
            "--cast",
            "signup_date=datetime",
            "-o",
            str(destination),
        ],
    )
    assert result.exit_code == 0
    ops = [step.op for step in load_recipe(destination).steps]
    assert ops.count("cast") == 2


def test_malformed_pair_option_is_rejected(messy_csv: Path, tmp_path: Path) -> None:
    """A missing '=' is a usage error with the expected form shown."""
    result = runner.invoke(
        app,
        ["clean", str(messy_csv), "--impute", "salary", "-o", str(tmp_path / "r.yaml")],
    )
    assert result.exit_code != 0


def test_no_input_produces_an_empty_recipe_without_prompting(
    tidy_csv: Path, tmp_path: Path
) -> None:
    """`--no-input` never blocks, even with nothing to do."""
    result = runner.invoke(
        app, ["clean", str(tidy_csv), "--no-input", "-o", str(tmp_path / "r.yaml")]
    )
    assert result.exit_code == 0
    assert "nothing to write" in result.output.lower()


# --- applying --------------------------------------------------------------


def _recipe_for(messy_csv: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "recipe.yaml"
    runner.invoke(
        app,
        [
            "clean",
            str(messy_csv),
            "--drop",
            "notes,country",
            "--dedupe",
            "--impute",
            "salary=median",
            "-o",
            str(destination),
        ],
    )
    return destination


def test_apply_writes_a_new_file(messy_csv: Path, tmp_path: Path) -> None:
    """Applying produces a new dataset; the input is untouched."""
    recipe = _recipe_for(messy_csv, tmp_path)
    output = tmp_path / "clean.csv"
    before = messy_csv.read_bytes()

    result = runner.invoke(
        app, ["apply", str(recipe), str(messy_csv), "-o", str(output)]
    )
    assert result.exit_code == 0
    assert output.exists()
    assert messy_csv.read_bytes() == before


def test_apply_reports_the_shape_change(messy_csv: Path, tmp_path: Path) -> None:
    """The user sees what the recipe did to the data's shape."""
    recipe = _recipe_for(messy_csv, tmp_path)
    result = runner.invoke(app, ["apply", str(recipe), str(messy_csv)])
    assert "→" in result.output


def test_apply_warns_about_fitting_on_everything(
    messy_csv: Path, tmp_path: Path
) -> None:
    """The leakage risk of fitting on a whole file is never hidden."""
    recipe = _recipe_for(messy_csv, tmp_path)
    result = runner.invoke(app, ["apply", str(recipe), str(messy_csv)])
    assert "michi export" in result.output


def test_apply_without_output_writes_nothing(messy_csv: Path, tmp_path: Path) -> None:
    """Without --out, apply reports but creates no file."""
    recipe = _recipe_for(messy_csv, tmp_path)
    before = set(tmp_path.iterdir())
    runner.invoke(app, ["apply", str(recipe), str(messy_csv)])
    assert set(tmp_path.iterdir()) == before


def test_apply_writes_parquet(messy_csv: Path, tmp_path: Path) -> None:
    """The output format follows the extension."""
    recipe = _recipe_for(messy_csv, tmp_path)
    output = tmp_path / "clean.parquet"
    runner.invoke(app, ["apply", str(recipe), str(messy_csv), "-o", str(output)])
    assert pd.read_parquet(output).shape[0] > 0


def test_apply_rejects_an_unknown_output_format(
    messy_csv: Path, tmp_path: Path
) -> None:
    """An unsupported extension names the ones that work."""
    recipe = _recipe_for(messy_csv, tmp_path)
    result = runner.invoke(
        app, ["apply", str(recipe), str(messy_csv), "-o", str(tmp_path / "out.xyz")]
    )
    assert result.exit_code == 2


def test_apply_to_mismatched_data_fails_loudly(
    messy_csv: Path, tidy_csv: Path, tmp_path: Path
) -> None:
    """A recipe applied to the wrong dataset is an error, not a surprise."""
    recipe = _recipe_for(messy_csv, tmp_path)
    result = runner.invoke(app, ["apply", str(recipe), str(tidy_csv)])
    assert result.exit_code == 2


def test_no_strict_allows_partial_application(
    messy_csv: Path, tidy_csv: Path, tmp_path: Path
) -> None:
    """`--no-strict` applies what still makes sense and reports the rest."""
    recipe = _recipe_for(messy_csv, tmp_path)
    result = runner.invoke(app, ["apply", str(recipe), str(tidy_csv), "--no-strict"])
    assert result.exit_code == 0
    assert "note:" in result.output


# --- exporting -------------------------------------------------------------


def test_export_writes_a_python_module(messy_csv: Path, tmp_path: Path) -> None:
    """Exported code lands as an ordinary Python file."""
    recipe = _recipe_for(messy_csv, tmp_path)
    output = tmp_path / "pipeline.py"
    result = runner.invoke(app, ["export", str(recipe), "-o", str(output)])
    assert result.exit_code == 0
    assert "def prepare(" in output.read_text(encoding="utf-8")


def test_export_prints_to_stdout_without_an_output(
    messy_csv: Path, tmp_path: Path
) -> None:
    """Without --out, the code goes to stdout so it can be piped."""
    recipe = _recipe_for(messy_csv, tmp_path)
    result = runner.invoke(app, ["export", str(recipe)])
    assert "def prepare(" in result.output


def test_export_states_that_michi_is_not_a_dependency(
    messy_csv: Path, tmp_path: Path
) -> None:
    """The exit door is advertised, not hidden."""
    recipe = _recipe_for(messy_csv, tmp_path)
    result = runner.invoke(app, ["export", str(recipe), "-o", str(tmp_path / "p.py")])
    assert "not a dependency" in result.output


def test_export_of_a_missing_recipe_fails_clearly(tmp_path: Path) -> None:
    """A missing recipe exits 2 with a readable message."""
    result = runner.invoke(app, ["export", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 2
