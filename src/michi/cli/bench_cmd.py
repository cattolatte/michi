"""The ``michi bench`` command.

Design Principles
-----------------
- ``--list-models`` prints the menu. michi never picks models for the user,
  so discovering the options must not require reading documentation.
- The preparation applied to raw columns is printed every run, because an
  assumption a user can read is not a decision michi made for them.
- Results are ranked, but the leader is never called "best" when the data
  cannot distinguish it from the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from michi.core.errors import MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table
from michi.core.manifest import RunManifest

__all__ = ["bench_command"]

_DEFAULT_MODELS = "linear,rf,hist-gbm"


def bench_command(
    data: Annotated[
        Path | None,
        typer.Argument(help="Dataset to benchmark on.", show_default=False),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Label column.", show_default=False),
    ] = None,
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated model names. A dummy baseline is always added.",
        ),
    ] = _DEFAULT_MODELS,
    list_models: Annotated[
        bool,
        typer.Option("--list-models", help="Print the model menu and exit."),
    ] = False,
    task: Annotated[
        str | None,
        typer.Option("--task", help="Force 'classification' or 'regression'."),
    ] = None,
    folds: Annotated[
        int, typer.Option("--cv", help="Number of cross-validation folds.")
    ] = 5,
    impute: Annotated[
        str,
        typer.Option("--impute", help="Numeric imputation: median, mean, or constant."),
    ] = "median",
    encode: Annotated[
        str,
        typer.Option("--encode", help="Categorical encoding: onehot or ordinal."),
    ] = "onehot",
    no_scale: Annotated[
        bool,
        typer.Option("--no-scale", help="Never standardise, even for linear models."),
    ] = False,
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Directory to write manifests into.")
    ] = Path("runs"),
    no_save: Annotated[
        bool, typer.Option("--no-save", help="Do not write run manifests.")
    ] = False,
    report_to: Annotated[
        Path | None,
        typer.Option("--report", help="Write an HTML report of the comparison here."),
    ] = None,
    open_report: Annotated[
        bool, typer.Option("--open", help="Open the HTML report in a browser.")
    ] = False,
    explain: Annotated[
        bool,
        typer.Option("--explain/--no-explain", help="Explain what each check means."),
    ] = False,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
    seed: Annotated[
        int, typer.Option("--seed", help="Seed for folds and model randomness.")
    ] = 0,
) -> None:
    """Train several models and report which are actually different.

    Cross-validates every model you name, always alongside a dummy baseline,
    and tests whether the differences between them survive the noise. Column
    preparation is fitted inside each fold, so a benchmark cannot leak.
    """
    console = Console()

    if list_models:
        _print_model_menu(console, task)
        raise typer.Exit()

    if data is None or target is None:
        msg = "michi bench needs a dataset and --target (or use --list-models)"
        raise typer.BadParameter(msg)

    try:
        from michi.bench import PreparationPolicy, run_benchmark
        from michi.report import render_benchmark

        table = load_table(data, sample_rows=sample, full=full, seed=seed)
        result = run_benchmark(
            table,
            target=target,
            models=tuple(name.strip() for name in models.split(",") if name.strip()),
            task=task,
            folds=folds,
            policy=PreparationPolicy(
                numeric_impute=impute, encode=encode, scale=not no_scale
            ),
            seed=seed,
        )
    except MichiError as err:
        Console(stderr=True).print(f"[bold red]error[/] {err}")
        raise typer.Exit(code=2) from err

    render_benchmark(result, console, explain=explain)

    written: list[Path] = []
    if not no_save:
        for manifest in result.manifests:
            destination = runs_dir / f"{manifest.run_id}.json"
            _write_manifest(manifest, destination)
        if result.manifests:
            written.append(runs_dir)
    if report_to is not None:
        from michi.report import render_benchmark_html

        report_to.parent.mkdir(parents=True, exist_ok=True)
        report_to.write_text(render_benchmark_html(result), encoding="utf-8")
        written.append(report_to)

    for destination in written:
        label = (
            f"{len(result.manifests)} manifests to {destination}"
            if destination == runs_dir
            else str(destination)
        )
        console.print(f"  [dim]wrote[/] {label}")
    if written:
        console.print()

    if open_report and report_to is not None:
        import webbrowser

        webbrowser.open(report_to.resolve().as_uri())


def _print_model_menu(console: Console, task: str | None) -> None:
    """Print the model catalogue — a menu, not a recommendation."""
    from rich import box
    from rich.table import Table

    from michi.bench import available_models

    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold dim",
        pad_edge=False,
        show_edge=False,
        title="Models michi can train — pick the ones you want to compare",
        title_style="bold",
        title_justify="left",
    )
    table.add_column("name", no_wrap=True)
    table.add_column("tasks", no_wrap=True)
    table.add_column("extra", no_wrap=True)
    table.add_column("what it is", overflow="fold")

    for entry in available_models(task):
        tasks = "both" if len(entry.tasks) == 2 else next(iter(entry.tasks))[:5]
        table.add_row(
            entry.name,
            tasks,
            entry.extra or "—",
            entry.summary,
        )
    console.print()
    console.print(table)
    console.print()


def _write_manifest(manifest: RunManifest, destination: Path) -> None:
    """Write a run manifest as formatted, UTF-8 JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
