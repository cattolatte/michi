"""The ``michi ensemble`` command — combine models, and check it was worth it.

Design Principles
-----------------
- **It renders as a benchmark because it is one.** The ensemble is
  cross-validated beside its own members, so the leaderboard answers the only
  question that matters: did combining beat the best single model by more than
  the noise?
- **michi picks no members.** The combination is the user's; michi neither
  searches for one nor drops a member for scoring poorly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from michi.cli.context import resolve_defaults
from michi.cli.errors import fail
from michi.core.errors import MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table

__all__ = ["ensemble_command"]


def ensemble_command(
    data: Annotated[
        Path | None,
        typer.Argument(help="Dataset. Falls back to `data` in michi.toml."),
    ] = None,
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Label column.")
    ] = None,
    models: Annotated[
        str,
        typer.Option("--models", "-m", help="Comma-separated members to combine."),
    ] = "linear,rf,hist-gbm",
    method: Annotated[
        str,
        typer.Option("--method", help="stack (meta-learner) or vote (average)."),
    ] = "stack",
    final: Annotated[
        str, typer.Option("--final", help="Meta-learner for stacking.")
    ] = "linear",
    recipe: Annotated[
        Path | None, typer.Option("--recipe", help="Cleaning recipe to apply first.")
    ] = None,
    cv: Annotated[int, typer.Option("--cv", help="Cross-validation folds.")] = 5,
    task: Annotated[
        str | None,
        typer.Option("--task", help="Force `classification` or `regression`."),
    ] = None,
    explain: Annotated[
        bool,
        typer.Option("--explain/--no-explain", help="Explain the numbers produced."),
    ] = False,
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", help="Where manifests are written.")
    ] = None,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Write run manifests.")
    ] = True,
    report: Annotated[
        Path | None, typer.Option("--report", help="Write an HTML report.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
) -> None:
    """Combine several models, and compare the result to each of them.

    The ensemble is cross-validated beside its own members and tested with the
    same corrected resampled t-test, so an ensemble that merely ties its best
    member is reported as a tie rather than as a win.
    """
    console = Console()
    defaults = resolve_defaults()
    resolved_seed = defaults.number("seed", seed) or 0

    try:
        from michi.bench.ensemble import run_ensemble
        from michi.report import render_benchmark

        resolved_data = defaults.required_data(data)
        table = load_table(
            resolved_data, sample_rows=sample, full=full, seed=resolved_seed
        )
        resolved_target, note = defaults.target_for(target, table.frame.columns)
        if note:
            console.print(f"  [dim]{note}[/]")
        if not resolved_target:
            msg = (
                "an ensemble needs a target column: pass --target, "
                "or set `target` in michi.toml"
            )
            raise MichiError(msg)

        loaded_recipe = None
        if recipe is not None:
            from michi.recipes import load_recipe

            loaded_recipe = load_recipe(recipe)

        members = tuple(item.strip() for item in models.split(",") if item.strip())
        result = run_ensemble(
            table,
            target=resolved_target,
            members=members,
            method=method,
            final=final,
            task=task,
            folds=cv,
            recipe=loaded_recipe,
            seed=resolved_seed,
        )
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    render_benchmark(result, console, explain=explain)

    if save:
        from michi.cli.bench_cmd import _write_manifest

        destination = defaults.path("runs_dir", runs_dir) or Path("runs")
        for manifest in result.manifests:
            _write_manifest(manifest, destination / f"{manifest.run_id}.json")
        console.print(
            f"  [dim]wrote {len(result.manifests)} manifest(s) to {destination}[/]\n"
        )

    if report is not None:
        from michi.report import render_benchmark_html

        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_benchmark_html(result), encoding="utf-8")
        console.print(f"  [dim]wrote {report}[/]\n")
