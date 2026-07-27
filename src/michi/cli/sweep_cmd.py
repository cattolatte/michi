"""The ``michi sweep`` command.

Design Principles
-----------------
- The grid is a file the user wrote; michi executes it and records every cell.
- Progress is printed as the sweep runs, because a grid that takes an hour
  must not look like a hang.
- Resumption is the default: re-running a sweep reuses valid recorded cells
  and does only what changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from michi.cli.errors import fail
from michi.core.errors import MichiError

__all__ = ["sweep_command"]


def sweep_command(
    plan_file: Annotated[
        Path, typer.Argument(help="Sweep plan (YAML).", show_default=False)
    ],
    sweep_dir: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Where cell manifests are written."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-run every cell, ignoring recorded results."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List the grid without running anything."),
    ] = False,
) -> None:
    """Run a grid of experiments: models × recipes × seeds.

    Every cell is identified by a hash of the data, recipe, model, seed, and
    folds that produce it, so an interrupted sweep resumes exactly where it
    stopped and changing one input re-runs only the cells it affects.
    """
    console = Console()
    try:
        from michi.sweep import load_plan, run_sweep

        plan = load_plan(plan_file)
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    cells = plan.cells()
    console.print()
    console.print(_header(plan, len(cells)))
    console.print()

    if dry_run:
        for index, cell in enumerate(cells, start=1):
            console.print(f"  [dim]{index:>3}[/]  {cell.label}")
        console.print(f"\n  [dim]{len(cells)} cells — nothing was run[/]\n")
        return

    def progress(index: int, total: int, cell: object, status: str) -> None:
        style = {"ran": "green", "cached": "dim", "failed": "red"}[status]
        label = getattr(cell, "label", str(cell))
        console.print(f"  [dim]{index:>3}/{total}[/]  [{style}]{status:<7}[/]  {label}")

    try:
        result = run_sweep(plan, sweep_dir=sweep_dir, force=force, progress=progress)
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    console.print()
    console.print(_summary(result))
    ranked = result.ranked()
    if ranked:
        console.print(_leaderboard(ranked))
    console.print(
        f"  [dim]wrote {len(result.outcomes)} cell manifests to "
        f"{result.sweep_dir}  ·  `michi report {result.sweep_dir}` renders them[/]\n"
    )


def _header(plan: object, size: int) -> Text:
    text = Text()
    text.append(" 道 ", style="bold red")
    text.append(" michi sweep", style="bold")
    text.append("  ·  ", style="dim")
    text.append(f"{size} cells", style="bold cyan")
    text.append("\n\n")
    text.append(
        f"  {getattr(plan, 'target', '')} · {getattr(plan, 'folds', 5)}-fold · "
        f"{len(getattr(plan, 'models', ()))} models × "
        f"{max(1, len(getattr(plan, 'recipes', ())))} recipes × "
        f"{len(getattr(plan, 'seeds', ()))} seeds",
        style="dim",
    )
    return text


def _summary(result: object) -> Text:
    counts = result.counts()  # type: ignore[attr-defined]
    text = Text()
    text.append("  ")
    text.append(f"{counts.get('ran', 0)} ran", style="green")
    text.append("  ·  ", style="dim")
    text.append(f"{counts.get('cached', 0)} reused", style="dim")
    if counts.get("failed"):
        text.append("  ·  ", style="dim")
        text.append(f"{counts['failed']} failed", style="red")
    text.append("  ·  ", style="dim")
    text.append(f"{result.duration_s:.1f}s", style="dim")  # type: ignore[attr-defined]
    text.append("\n")
    return text


def _leaderboard(ranked: tuple[object, ...]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold dim",
        pad_edge=False,
        show_edge=False,
        title="  Cells, best first",
        title_style="bold",
        title_justify="left",
    )
    table.add_column("  model", no_wrap=True)
    table.add_column("recipe", no_wrap=True)
    table.add_column("seed", justify="right", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)

    for outcome in ranked[:15]:
        cell = outcome.cell  # type: ignore[attr-defined]
        recipe = Path(cell.recipe).stem if cell.recipe else "—"
        score = outcome.score  # type: ignore[attr-defined]
        table.add_row(
            f"  {cell.model}",
            recipe,
            str(cell.seed),
            f"{score:.4g}" if score is not None else "—",
        )
    return table
