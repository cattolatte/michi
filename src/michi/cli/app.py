"""The michi command-line application.

Design Principles
-----------------
- The CLI layer is thin: it parses arguments, calls domain modules, and
  renders results. No domain logic lives here.
- Every capability is reachable non-interactively; interactive flows are
  sugar over flags, never the only path.
- Command modules are imported eagerly only when cheap; heavy domain imports
  happen inside the command body so ``michi --version`` stays instant.
- Bare ``michi`` will become the interactive console in v0.5; until then it
  shows help.
"""

from __future__ import annotations

import platform
import sys
from typing import Annotated

import typer
from rich.console import Console

from michi import __version__
from michi.cli.bench_cmd import bench_command
from michi.cli.clean_cmd import apply_command, clean_command, export_command
from michi.cli.eval_cmd import eval_command
from michi.cli.fit_cmd import fit_command, predict_command
from michi.cli.inspect_cmd import inspect_command
from michi.cli.report_cmd import report_command
from michi.cli.sweep_cmd import sweep_command
from michi.cli.tune_cmd import tune_command
from michi.cli.ui_cmd import ui_command

app = typer.Typer(
    name="michi",
    help="Michi (道) — a local-first ML workbench. "
    "Automate implementation, never judgement.",
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode="rich",
)
_console = Console()

app.command("inspect")(inspect_command)
app.command("eval")(eval_command)
app.command("bench")(bench_command)
app.command("report")(report_command)
app.command("clean")(clean_command)
app.command("apply")(apply_command)
app.command("export")(export_command)
app.command("tune")(tune_command)
app.command("fit")(fit_command)
app.command("predict")(predict_command)
app.command("sweep")(sweep_command)
app.command("ui")(ui_command)


def _print_version(value: bool) -> None:
    if value:
        _console.print(f"michi {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    context: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the michi version and exit.",
        ),
    ] = False,
) -> None:
    """Michi (道) — the path through repetitive ML work.

    Run with no arguments to open the interactive console.
    """
    if context.invoked_subcommand is not None:
        return

    # Bare `michi` opens the console when a terminal is attached, and prints
    # help otherwise, so piping or scripting `michi` never hangs on a prompt.
    if not _console.is_terminal:
        _console.print(context.get_help())
        raise typer.Exit()

    from michi.console import run_console

    raise typer.Exit(code=run_console())


@app.command()
def plugins() -> None:
    """List installed michi plugins, working or not."""
    from rich import box
    from rich.table import Table

    from michi.plugins import installed_plugins

    records = installed_plugins()
    if not records:
        _console.print(
            "\n  [dim]no plugins installed[/]\n\n"
            "  michi discovers entry points in two groups:\n"
            "    [cyan]michi.models[/]    add algorithms to the bench catalogue\n"
            "    [cyan]michi.adapters[/]  add ways to load a model for eval\n"
        )
        return

    table = Table(
        box=box.SIMPLE_HEAD, header_style="bold dim", pad_edge=False, show_edge=False
    )
    table.add_column("group", no_wrap=True)
    table.add_column("name", no_wrap=True)
    table.add_column("from", no_wrap=True)
    table.add_column("status", overflow="fold")
    for record in records:
        status = (
            "[green]loaded[/]" if record.loaded else f"[red]failed[/] {record.error}"
        )
        table.add_row(record.group, record.name, record.distribution, status)
    _console.print()
    _console.print(table)
    _console.print()


@app.command()
def info() -> None:
    """Show version and environment information."""
    from michi.core.config import find_config, load_defaults
    from michi.core.io import supported_formats

    _console.print(f"michi     {__version__}")
    _console.print(f"python    {sys.version.split()[0]}")
    _console.print(f"platform  {platform.platform()}")
    _console.print(f"formats   {', '.join(supported_formats())}")

    config = find_config()
    if config is None:
        _console.print("config    [dim]none (no michi.toml found)[/]")
        return
    _console.print(f"config    {config}")
    defaults = load_defaults()
    for key, value in (
        ("data", defaults.data),
        ("target", defaults.target),
        ("recipe", defaults.recipe),
        ("runs_dir", defaults.runs_dir),
        ("models", defaults.models),
        ("seed", defaults.seed),
        ("cv", defaults.cv),
    ):
        if value is not None:
            _console.print(f"  [dim]{key}[/] = {value}")
