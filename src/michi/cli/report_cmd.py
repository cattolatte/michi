"""The ``michi report`` command.

Design Principles
-----------------
- The runs directory is the only input: no index, no database, no state that
  can drift from the files on disk.
- Every output format renders the same artifacts, so a paper table and a
  browser page can never disagree.
- Runs are only ever compared within a dataset and target, because metrics
  across different data are not on the same scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from michi.cli.context import resolve_defaults
from michi.cli.errors import fail
from michi.core.errors import MichiError

__all__ = ["report_command"]


def report_command(
    source: Annotated[
        Path | None,
        typer.Argument(
            help="Runs directory or a single manifest file. "
            "Falls back to `runs_dir` in michi.toml."
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write the report here instead of stdout."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="html, markdown, or latex."),
    ] = "html",
    open_report: Annotated[
        bool, typer.Option("--open", help="Open the written report in a browser.")
    ] = False,
) -> None:
    """Render recorded runs as a report.

    Reads the manifests written by 'michi eval' and 'michi bench', groups them
    by dataset and target, and renders them as a self-contained HTML page,
    Markdown, or a LaTeX table ready to paste into a paper.
    """
    console = Console()
    source = resolve_defaults().path("runs_dir", source) or Path("runs")
    try:
        from michi.report import (
            render_runs_html,
            render_runs_latex,
            render_runs_markdown,
            render_runs_terminal,
        )
        from michi.report.runs import group_runs, load_manifests

        manifests = load_manifests(source)
        groups = group_runs(manifests)
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    renderers = {
        "html": render_runs_html,
        "markdown": render_runs_markdown,
        "md": render_runs_markdown,
        "latex": render_runs_latex,
        "tex": render_runs_latex,
    }
    if output_format not in renderers:
        known = ", ".join(sorted(set(renderers)))
        msg = f"unknown format {output_format!r}; expected one of: {known}"
        raise typer.BadParameter(msg)

    if output is None and output_format == "html":
        render_runs_terminal(groups, console)
        console.print(
            "  [dim]pass --out report.html to write the full report, "
            "or --format markdown to print it here[/]\n"
        )
        return

    rendered = renderers[output_format](groups)
    if output is None:
        console.print(rendered, markup=False, highlight=False)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    console.print(f"  [dim]wrote[/] {output}\n")

    if open_report:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())
