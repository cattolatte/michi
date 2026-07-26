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
from michi.cli.inspect_cmd import inspect_command
from michi.cli.report_cmd import report_command

app = typer.Typer(
    name="michi",
    help="Michi (道) — a local-first ML workbench. "
    "Automate implementation, never judgement.",
    no_args_is_help=True,
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


def _print_version(value: bool) -> None:
    if value:
        _console.print(f"michi {__version__}")
        raise typer.Exit()


@app.callback()
def main(
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
    """Michi (道) — the path through repetitive ML work."""


@app.command()
def info() -> None:
    """Show version and environment information."""
    from michi.core.io import supported_formats

    _console.print(f"michi     {__version__}")
    _console.print(f"python    {sys.version.split()[0]}")
    _console.print(f"platform  {platform.platform()}")
    _console.print(f"formats   {', '.join(supported_formats())}")
