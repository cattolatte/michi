"""Tests for the CLI skeleton."""

from __future__ import annotations

from typer.testing import CliRunner

from michi import __version__
from michi.cli.app import app

runner = CliRunner()

# --- version and info ------------------------------------------------------


def test_version_flag_prints_version() -> None:
    """`michi --version` prints the package version and exits cleanly."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_info_reports_environment() -> None:
    """`michi info` runs fully offline and reports the version."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "michi" in result.output
    assert __version__ in result.output


# --- bare invocation -------------------------------------------------------


def test_bare_invocation_shows_help() -> None:
    """Bare `michi` shows help until the console ships (PLAN.md §15, v0.5)."""
    result = runner.invoke(app, [])
    assert "Usage" in result.output
