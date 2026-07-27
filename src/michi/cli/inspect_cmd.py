"""The ``michi inspect`` command.

Design Principles
-----------------
- The command parses arguments, calls the domain packages, and renders. It
  contains no profiling logic of its own.
- Nothing is written unless the user asked for it: ``inspect`` prints to the
  terminal and only creates files when ``--html`` or ``--json`` is given.
- Every option has a non-interactive form and machine-readable output, so the
  same command serves a human, a CI job, and a script identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from michi.cli.context import resolve_defaults
from michi.cli.errors import fail
from michi.core.artifacts import DatasetProfile, Severity
from michi.core.errors import MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table
from michi.inspection import profile_table
from michi.report import render_profile, render_profile_html

__all__ = ["inspect_command"]


def inspect_command(
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Dataset to profile (.csv, .tsv, .parquet, or .xlsx). "
            "Falls back to `data` in michi.toml.",
            show_default=False,
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            "-t",
            help="Label column; enables class-imbalance and leakage checks.",
        ),
    ] = None,
    html: Annotated[
        Path | None,
        typer.Option("--html", help="Write a self-contained HTML report here."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the profile artifact as JSON here."),
    ] = None,
    open_report: Annotated[
        bool,
        typer.Option("--open", help="Open the HTML report in a browser."),
    ] = False,
    explain: Annotated[
        bool,
        typer.Option(
            "--explain/--no-explain",
            help="Print what each finding means and which options exist.",
        ),
    ] = False,
    sample: Annotated[
        int,
        typer.Option("--sample", help="Rows to keep when a large file is sampled."),
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool,
        typer.Option("--full", help="Read every row, however large the file."),
    ] = False,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for reproducible sampling.")
    ] = None,
    max_columns: Annotated[
        int | None,
        typer.Option("--max-columns", help="Truncate the column table after N rows."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Print only findings, not the full table."),
    ] = False,
    fail_on: Annotated[
        str | None,
        typer.Option(
            "--fail-on",
            help="Exit non-zero if any finding reaches this severity "
            "(high, warn, or info). For CI gates.",
        ),
    ] = None,
) -> None:
    """Profile a dataset and explain what stands out.

    Reports column kinds, missing values, duplicates, cardinality, skew,
    outliers and redundancy. Naming a target additionally checks class balance
    and flags possible leakage. michi describes what it finds; deciding what
    to do about it is your call.
    """
    console = Console()
    defaults = resolve_defaults()
    seed = defaults.number("seed", seed) or 0
    try:
        resolved = defaults.required_data(path)
        table = load_table(resolved, sample_rows=sample, full=full, seed=seed)
        resolved_target, note = defaults.target_for(target, table.frame.columns)
        if note:
            console.print(f"  [dim]{note}[/]")
        profile = profile_table(table, target=resolved_target)
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    render_profile(
        profile,
        console,
        explain=explain,
        max_columns=0 if quiet else max_columns,
    )

    written: list[Path] = []
    if json_out is not None:
        _write_json(profile, json_out)
        written.append(json_out)
    if html is not None:
        _write_html(profile, html)
        written.append(html)
    for destination in written:
        console.print(f"  [dim]wrote[/] {destination}")
    if written:
        console.print()

    if open_report and html is not None:
        import webbrowser

        webbrowser.open(html.resolve().as_uri())

    if fail_on is not None:
        raise typer.Exit(code=_exit_code_for(profile, fail_on))


def _write_json(profile: DatasetProfile, destination: Path) -> None:
    """Write the profile artifact as formatted, UTF-8 JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(profile.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_html(profile: DatasetProfile, destination: Path) -> None:
    """Write the self-contained HTML report."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_profile_html(profile), encoding="utf-8")


def _exit_code_for(profile: DatasetProfile, fail_on: str) -> int:
    """Return 1 when any finding is at least as severe as ``fail_on``."""
    try:
        threshold = Severity(fail_on.lower())
    except ValueError as err:
        msg = f"--fail-on must be one of: high, warn, info (got {fail_on!r})"
        raise typer.BadParameter(msg) from err
    return int(
        any(finding.severity.rank <= threshold.rank for finding in profile.findings)
    )
