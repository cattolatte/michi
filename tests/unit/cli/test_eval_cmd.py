"""Tests for the `michi eval` command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from michi.cli.app import app
from michi.core.manifest import RunManifest

runner = CliRunner()

# --- terminal output -------------------------------------------------------


def test_eval_reports_metrics_and_baselines(
    classification_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """The default run prints metrics beside the trivial baselines."""
    model_path, data_path = classification_data
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--no-save",
            "--bootstrap",
            "0",
        ],
    )
    assert result.exit_code == 0
    assert "balanced_accuracy" in result.output
    assert "most_frequent" in result.output


def test_eval_shows_a_confusion_matrix(
    classification_data: tuple[Path, Path],
) -> None:
    """Classification runs show where predictions went wrong."""
    model_path, data_path = classification_data
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--no-save",
            "--bootstrap",
            "0",
        ],
    )
    assert "Confusion" in result.output


def test_eval_explains_its_checks(classification_data: tuple[Path, Path]) -> None:
    """`--explain` adds meaning and options for each check raised."""
    model_path, data_path = classification_data
    explained = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--no-save",
            "--bootstrap",
            "0",
            "--explain",
        ],
    )
    assert "Options:" in explained.output


# --- manifests -------------------------------------------------------------


def test_eval_writes_a_manifest_by_default(
    classification_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Every run is recorded, because an unrepeatable result is an anecdote."""
    model_path, data_path = classification_data
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--runs-dir",
            str(runs),
            "--bootstrap",
            "0",
        ],
    )
    assert result.exit_code == 0
    written = list(runs.glob("*.json"))
    assert len(written) == 1


def test_written_manifest_rebuilds(
    classification_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """The manifest round-trips back into an artifact object."""
    model_path, data_path = classification_data
    destination = tmp_path / "run.json"
    runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--no-save",
            "--json",
            str(destination),
            "--bootstrap",
            "0",
        ],
    )
    manifest = RunManifest.from_dict(
        json.loads(destination.read_text(encoding="utf-8"))
    )
    assert manifest.task == "classification"
    assert manifest.metrics


def test_no_save_writes_nothing(
    classification_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """`--no-save` leaves the filesystem untouched."""
    model_path, data_path = classification_data
    runs = tmp_path / "runs"
    runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--runs-dir",
            str(runs),
            "--no-save",
            "--bootstrap",
            "0",
        ],
    )
    assert not runs.exists()


# --- gates and failures ----------------------------------------------------


def test_fail_under_passes_when_the_metric_is_met(
    classification_data: tuple[Path, Path],
) -> None:
    """A satisfied threshold exits zero."""
    model_path, data_path = classification_data
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--no-save",
            "--bootstrap",
            "0",
            "--fail-under",
            "accuracy=0.1",
        ],
    )
    assert result.exit_code == 0
    assert "gate passed" in result.output


def test_fail_under_fails_when_the_metric_is_missed(
    classification_data: tuple[Path, Path],
) -> None:
    """An unmet threshold exits non-zero, for CI."""
    model_path, data_path = classification_data
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--no-save",
            "--bootstrap",
            "0",
            "--fail-under",
            "accuracy=0.999",
        ],
    )
    assert result.exit_code == 1
    assert "gate failed" in result.output


def test_fail_under_respects_metric_direction(
    regression_data: tuple[Path, Path],
) -> None:
    """For error metrics, lower passes — the direction is not guessed."""
    model_path, data_path = regression_data
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "value",
            "--no-save",
            "--bootstrap",
            "0",
            "--fail-under",
            "rmse=100",
        ],
    )
    assert result.exit_code == 0


def test_unknown_metric_in_gate_lists_the_available_ones(
    classification_data: tuple[Path, Path],
) -> None:
    """A typo in the gate names what the run actually recorded."""
    model_path, data_path = classification_data
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "churned",
            "--no-save",
            "--bootstrap",
            "0",
            "--fail-under",
            "fscore=0.5",
        ],
    )
    assert result.exit_code != 0


def test_missing_model_exits_with_error_code(tmp_path: Path) -> None:
    """A missing model exits 2 with a readable message."""
    data = tmp_path / "data.csv"
    data.write_text("a,label\n1,0\n2,1\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "eval",
            str(tmp_path / "absent.pkl"),
            str(data),
            "--target",
            "label",
            "--no-save",
        ],
    )
    assert result.exit_code == 2


def test_regression_model_reports_regression_metrics(
    regression_data: tuple[Path, Path],
) -> None:
    """A regressor is detected and scored with regression metrics."""
    model_path, data_path = regression_data
    result = runner.invoke(
        app,
        [
            "eval",
            str(model_path),
            str(data_path),
            "--target",
            "value",
            "--no-save",
            "--bootstrap",
            "0",
        ],
    )
    assert "rmse" in result.output
    assert "r2" in result.output
