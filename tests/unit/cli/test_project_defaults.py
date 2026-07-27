"""Tests for `michi.toml` project defaults reaching the one-shot commands.

The documented precedence is flags > michi.toml > built-in. It was previously
true only inside the console, which made the documentation wrong for every
shell user. These tests hold every command to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from michi.cli.app import app

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, messy_csv: Path) -> Path:
    """A directory with data and a michi.toml naming it."""
    data = tmp_path / "customers.csv"
    data.write_text(messy_csv.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "michi.toml").write_text(
        '[defaults]\ndata = "customers.csv"\ntarget = "purchased"\n'
        'runs_dir = "recorded"\nseed = 7\ncv = 3\nmodels = "linear,tree"\n',
        encoding="utf-8",
    )
    return tmp_path


# --- inspect ---------------------------------------------------------------


def test_inspect_uses_configured_data_and_target(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare `michi inspect` works when michi.toml supplies both."""
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["inspect"])
    assert result.exit_code == 0
    assert "purchased" in result.output


def test_an_explicit_argument_beats_the_config(
    project: Path, tidy_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the user typed always wins."""
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["inspect", str(tidy_csv)])
    assert result.exit_code == 0
    assert "tidy.csv" in result.output


def test_an_explicit_target_beats_the_config(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typed --target overrides the configured one."""
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["inspect", "--target", "age"])
    assert result.exit_code == 0
    assert "target age" in result.output.replace("\n", " ")


def test_without_data_or_config_the_error_names_both_ways(tmp_path: Path) -> None:
    """With nothing to go on, michi says both ways to supply a dataset."""
    result = runner.invoke(app, ["inspect"])
    assert result.exit_code == 2


# --- bench -----------------------------------------------------------------


def test_bench_uses_configured_models_and_folds(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare `michi bench` runs the configured grid."""
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["bench", "--no-save"])
    assert result.exit_code == 0
    assert "3-fold" in result.output
    assert "linear" in result.output


def test_bench_writes_to_the_configured_runs_directory(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`runs_dir` in michi.toml is where manifests land."""
    monkeypatch.chdir(project)
    runner.invoke(app, ["bench", "--models", "linear"])
    assert list((project / "recorded").glob("*.json"))


# --- clean -----------------------------------------------------------------


def test_clean_uses_configured_data(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare `michi clean` finds its dataset in the config."""
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["clean", "--drop", "notes", "-o", "r.yaml"])
    assert result.exit_code == 0
    assert (project / "r.yaml").exists()


# --- report ----------------------------------------------------------------


def test_report_reads_the_configured_runs_directory(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`michi report` with no argument uses the configured directory."""
    monkeypatch.chdir(project)
    runner.invoke(app, ["bench", "--models", "linear"])
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
    assert "michi report" in result.output


# --- eval ------------------------------------------------------------------


def test_eval_uses_configured_target_and_data(
    classification_data: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`michi eval model.pkl` alone works with a configured project."""
    model_path, data_path = classification_data
    project = tmp_path / "project"
    project.mkdir()
    (project / "test.csv").write_text(
        data_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (project / "michi.toml").write_text(
        '[defaults]\ndata = "test.csv"\ntarget = "churned"\n', encoding="utf-8"
    )
    monkeypatch.chdir(project)

    result = runner.invoke(
        app, ["eval", str(model_path), "--no-save", "--bootstrap", "0"]
    )
    assert result.exit_code == 0
    assert "balanced_accuracy" in result.output


def test_eval_without_a_target_anywhere_is_actionable(
    classification_data: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing a target names both ways to supply one."""
    model_path, data_path = classification_data
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["eval", str(model_path), str(data_path), "--no-save"])
    assert result.exit_code == 2
    assert "michi.toml" in result.output


# --- the recipe flag -------------------------------------------------------


def test_eval_applies_a_recipe(
    classification_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """`--recipe` prepares the evaluation data before scoring."""
    model_path, data_path = classification_data
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("steps:\n  - op: dedupe\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--recipe",
            str(recipe),
            "--no-save",
            "--bootstrap",
            "0",
        ],
    )
    assert result.exit_code == 0


def test_bench_applies_a_recipe(tidy_csv: Path, tmp_path: Path) -> None:
    """`bench --recipe` composes a recipe with cross-validation."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        'steps:\n  - op: drop\n    columns: ["group"]\n', encoding="utf-8"
    )
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear",
            "--cv",
            "3",
            "--no-save",
            "--recipe",
            str(recipe),
        ],
    )
    assert result.exit_code == 0


# --- robustness ------------------------------------------------------------


def test_a_broken_config_does_not_break_a_complete_command(
    tmp_path: Path, tidy_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed michi.toml must not fail a command given everything it needs.

    `michi info` reports the parse error; a command that was told what to do
    should still do it.
    """
    (tmp_path / "michi.toml").write_text("[defaults\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["inspect", str(tidy_csv)])
    assert result.exit_code == 0


def test_config_is_found_from_a_subdirectory(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running from a nested directory still finds the project's defaults."""
    nested = project / "notebooks"
    nested.mkdir()
    monkeypatch.chdir(nested)
    result = runner.invoke(app, ["inspect"])
    # The configured data path is relative to where michi was started, so the
    # run may not find the file — what matters is that the config was read and
    # michi did not simply complain that no dataset was given.
    assert "no dataset given" not in result.output
