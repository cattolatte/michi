"""Tests for the `michi inspect` command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from michi.cli.app import app

runner = CliRunner()

# --- terminal output -------------------------------------------------------


def test_inspect_reports_shape_and_findings(messy_csv: Path) -> None:
    """The default run prints the dataset shape and its findings."""
    result = runner.invoke(app, ["inspect", str(messy_csv)])
    assert result.exit_code == 0
    assert "120 rows" in result.output
    assert "Findings" in result.output


def test_inspect_marks_the_target_column(messy_csv: Path) -> None:
    """Naming a target surfaces it in the summary."""
    result = runner.invoke(app, ["inspect", str(messy_csv), "--target", "purchased"])
    assert result.exit_code == 0
    assert "purchased" in result.output


def test_explain_adds_meaning_and_options(messy_csv: Path) -> None:
    """`--explain` prints what findings mean and which options exist."""
    plain = runner.invoke(app, ["inspect", str(messy_csv)])
    explained = runner.invoke(app, ["inspect", str(messy_csv), "--explain"])
    assert "Options:" not in plain.output
    assert "Options:" in explained.output
    assert len(explained.output) > len(plain.output)


def test_explanations_offer_options_rather_than_advice(messy_csv: Path) -> None:
    """michi never tells the user what it recommends."""
    result = runner.invoke(app, ["inspect", str(messy_csv), "--explain"])
    lowered = result.output.lower()
    assert "we recommend" not in lowered
    assert "you should" not in lowered


# --- artifacts -------------------------------------------------------------


def test_json_output_is_a_valid_profile(messy_csv: Path, tmp_path: Path) -> None:
    """`--json` writes a parseable profile artifact with a schema version."""
    destination = tmp_path / "profile.json"
    result = runner.invoke(app, ["inspect", str(messy_csv), "--json", str(destination)])
    assert result.exit_code == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"]
    assert payload["n_rows"] == 120
    assert len(payload["columns"]) == payload["n_columns"]


def test_json_artifact_rebuilds_into_a_profile(messy_csv: Path, tmp_path: Path) -> None:
    """The written artifact round-trips back into a profile object."""
    from michi.core.artifacts import DatasetProfile

    destination = tmp_path / "profile.json"
    runner.invoke(app, ["inspect", str(messy_csv), "--json", str(destination)])
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert DatasetProfile.from_dict(payload).n_rows == 120


def test_html_output_is_written(messy_csv: Path, tmp_path: Path) -> None:
    """`--html` writes a complete HTML document."""
    destination = tmp_path / "report.html"
    result = runner.invoke(app, ["inspect", str(messy_csv), "--html", str(destination)])
    assert result.exit_code == 0
    assert destination.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_nothing_is_written_unless_requested(messy_csv: Path, tmp_path: Path) -> None:
    """A plain inspect run creates no files."""
    before = set(tmp_path.iterdir())
    runner.invoke(app, ["inspect", str(messy_csv)])
    assert set(tmp_path.iterdir()) == before


def test_output_directories_are_created(messy_csv: Path, tmp_path: Path) -> None:
    """Writing into a missing directory creates it rather than failing."""
    destination = tmp_path / "nested" / "deep" / "profile.json"
    result = runner.invoke(app, ["inspect", str(messy_csv), "--json", str(destination)])
    assert result.exit_code == 0
    assert destination.exists()


# --- exit codes ------------------------------------------------------------


def test_missing_file_exits_with_error_code(tmp_path: Path) -> None:
    """A missing dataset exits non-zero with a readable message."""
    result = runner.invoke(app, ["inspect", str(tmp_path / "absent.csv")])
    assert result.exit_code == 2


def test_fail_on_high_gates_a_problematic_dataset(messy_csv: Path) -> None:
    """`--fail-on high` exits non-zero when severe findings exist."""
    result = runner.invoke(app, ["inspect", str(messy_csv), "--fail-on", "high"])
    assert result.exit_code == 1


def test_fail_on_high_passes_a_clean_dataset(tidy_csv: Path) -> None:
    """`--fail-on high` exits zero when nothing severe was found."""
    result = runner.invoke(app, ["inspect", str(tidy_csv), "--fail-on", "high"])
    assert result.exit_code == 0


def test_invalid_fail_on_value_is_rejected(messy_csv: Path) -> None:
    """An unknown severity is a usage error, not a crash."""
    result = runner.invoke(app, ["inspect", str(messy_csv), "--fail-on", "critical"])
    assert result.exit_code != 0


# --- options ---------------------------------------------------------------


def test_seed_is_accepted_and_run_is_deterministic(messy_csv: Path) -> None:
    """Two runs with the same seed produce the same output."""
    first = runner.invoke(app, ["inspect", str(messy_csv), "--seed", "3"])
    second = runner.invoke(app, ["inspect", str(messy_csv), "--seed", "3"])
    assert first.output == second.output


def test_quiet_suppresses_the_column_table(messy_csv: Path) -> None:
    """`--quiet` keeps findings but drops the per-column detail."""
    full = runner.invoke(app, ["inspect", str(messy_csv)])
    quiet = runner.invoke(app, ["inspect", str(messy_csv), "--quiet"])
    assert "Findings" in quiet.output
    assert len(quiet.output) < len(full.output)
