"""``michi diff`` — has this data changed since the profile you trusted?

Design Principles
-----------------
- **Either side may be a profile or a dataset.** A committed
  ``profile.json`` is the natural baseline — it is small, diffable, and
  already in the repository — but comparing two CSVs directly has to work too,
  because that is what someone reaches for first.
- **A gate, like ``inspect``.** ``--fail-on high`` exits non-zero, so the same
  command that a person runs by hand is the one CI runs nightly.
- **It reports; it does not repair.** Whether a shifted column matters depends
  on what it feeds and what the model is for, and michi knows neither.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from michi.cli.errors import fail
from michi.core.artifacts import DatasetProfile, Severity
from michi.core.errors import DataError, MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table
from michi.report.terminal import severity_style

__all__ = ["diff_command"]

_LEVELS = {"high": Severity.HIGH, "warn": Severity.WARN, "info": Severity.INFO}


def diff_command(
    baseline: Annotated[
        Path,
        typer.Argument(
            help="What the data used to be: a profile.json or a dataset.",
            show_default=False,
        ),
    ],
    current: Annotated[
        Path,
        typer.Argument(
            help="What it is now: a profile.json or a dataset.", show_default=False
        ),
    ],
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Label column, if profiling.")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the comparison here.")
    ] = None,
    fail_on: Annotated[
        str | None,
        typer.Option("--fail-on", help="Exit non-zero at this severity: high|warn."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for sampling a large file.")
    ] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
) -> None:
    """Compare two datasets, or a dataset against a profile it should match.

    Reports schema changes, missingness, distribution shift, and new
    categories — the ways data stops resembling what a model was trained on.
    Either side may be a committed profile.json or a raw data file.
    """
    console = Console()
    from michi.cli.context import resolve_defaults

    resolved_seed = resolve_defaults().number("seed", seed) or 0
    try:
        from michi.inspection.drift import compare_profiles

        before = _as_profile(
            baseline, target=target, sample=sample, full=full, seed=resolved_seed
        )
        after = _as_profile(
            current, target=target, sample=sample, full=full, seed=resolved_seed
        )
        report = compare_profiles(before, after)
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    _render(console, report, baseline=baseline, current=current)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"  [dim]wrote[/] {json_out}\n")

    if fail_on is not None:
        raise typer.Exit(code=_gate(report, fail_on, console))


def _as_profile(
    path: Path, *, target: str | None, sample: int, full: bool, seed: int
) -> DatasetProfile:
    """Read a side of the comparison, profiling it first if it is raw data."""
    if not path.exists():
        msg = f"no such file: {path}"
        raise DataError(msg)

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            msg = f"could not parse {path.name} as JSON: {err}"
            raise DataError(msg) from err
        if "columns" not in payload:
            msg = (
                f"{path.name} is not a michi profile — it has no `columns`. "
                "Write one with `michi inspect data.csv --json profile.json`."
            )
            raise DataError(msg)
        return DatasetProfile.from_dict(payload)

    from michi.inspection import profile_table

    table = load_table(path, sample_rows=sample, full=full, seed=seed)
    resolved = target if target and target in table.frame.columns else None
    return profile_table(table, target=resolved)


def _render(console: Console, report: object, *, baseline: Path, current: Path) -> None:
    """Print what moved, most severe first."""
    header = Text()
    header.append(" 道 ", style="bold red")
    header.append(" michi diff", style="bold")
    console.print()
    console.print(header)
    console.print()

    summary = Text(style="dim")
    summary.append(
        f"{baseline.name} → {current.name}   ·   "
        f"{report.baseline_rows:,} → {report.current_rows:,} rows"  # type: ignore[attr-defined]
    )
    console.print(Padding(summary, (0, 0, 1, 2)))

    findings = report.findings  # type: ignore[attr-defined]
    if not findings:
        console.print(
            Padding(
                Text(
                    "Nothing moved — the data still matches the baseline.",
                    style="green",
                ),
                (0, 0, 1, 2),
            )
        )
        return

    table = Table(
        box=box.SIMPLE_HEAD, show_header=False, pad_edge=False, show_edge=False
    )
    table.add_column("severity", no_wrap=True)
    table.add_column("columns", no_wrap=True)
    table.add_column("what changed", overflow="fold")
    for finding in findings:
        where = ", ".join(finding.columns[:3]) or "dataset"
        table.add_row(
            Text(finding.severity.value, style=severity_style(finding.severity)),
            Text(where, style="bold"),
            Text(finding.summary),
        )
    console.print(Padding(table, (0, 0, 1, 2)))
    console.print(
        Padding(
            Text(
                "What a shift means depends on what the column feeds. michi "
                "reports the\nmovement; whether it matters is yours.",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )


def _gate(report: object, fail_on: str, console: Console) -> int:
    """Exit non-zero when something at or above `fail_on` moved."""
    level = _LEVELS.get(fail_on.lower())
    if level is None:
        fail(f"--fail-on expects high or warn (got {fail_on!r})")
        return 2

    order = {Severity.HIGH: 0, Severity.WARN: 1, Severity.INFO: 2}
    triggered = [
        item
        for item in report.findings  # type: ignore[attr-defined]
        if order[item.severity] <= order[level]
    ]
    if not triggered:
        return 0
    console.print(f"  [red]{len(triggered)} finding(s) at or above {level.value}[/]\n")
    return 1
