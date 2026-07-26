"""The ``michi eval`` command.

Design Principles
-----------------
- The command parses arguments, calls the domain packages, and renders. It
  contains no evaluation logic of its own.
- Every run writes a manifest by default: an evaluation you cannot reproduce
  is an anecdote, and the artifact costs nothing.
- ``--fail-under`` makes the same command a CI gate without a second code
  path.
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
from michi.evaluation import evaluate_model
from michi.evaluation.metrics import BOOTSTRAP_SAMPLES
from michi.report import render_evaluation

__all__ = ["eval_command"]


def eval_command(
    model: Annotated[
        str,
        typer.Argument(
            help="Model to evaluate: a pickle/joblib file, or 'mymodule:my_model'.",
            show_default=False,
        ),
    ],
    data: Annotated[
        Path,
        typer.Argument(help="Evaluation dataset.", show_default=False),
    ],
    target: Annotated[
        str,
        typer.Option("--target", "-t", help="Label column.", show_default=False),
    ],
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            help="Force 'classification' or 'regression' instead of inferring.",
        ),
    ] = None,
    features: Annotated[
        str | None,
        typer.Option(
            "--features",
            help="Comma-separated columns to pass to the model "
            "(default: every column but the target).",
        ),
    ] = None,
    slice_by: Annotated[
        str | None,
        typer.Option(
            "--slice",
            help="Comma-separated columns to score subgroups over "
            "(default: low-cardinality columns).",
        ),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory to write the run manifest into."),
    ] = Path("runs"),
    no_save: Annotated[
        bool,
        typer.Option("--no-save", help="Do not write a run manifest."),
    ] = False,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Also write the manifest here."),
    ] = None,
    explain: Annotated[
        bool,
        typer.Option(
            "--explain/--no-explain",
            help="Print what each check means and which options exist.",
        ),
    ] = False,
    bootstrap: Annotated[
        int,
        typer.Option(
            "--bootstrap",
            help="Resamples for confidence intervals; 0 disables them.",
        ),
    ] = BOOTSTRAP_SAMPLES,
    sample: Annotated[
        int,
        typer.Option("--sample", help="Rows to keep when a large file is sampled."),
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
    seed: Annotated[
        int, typer.Option("--seed", help="Seed for sampling and resampling.")
    ] = 0,
    fail_under: Annotated[
        str | None,
        typer.Option(
            "--fail-under",
            help="Exit non-zero unless a metric reaches a value, "
            "e.g. 'f1=0.85'. For CI gates.",
        ),
    ] = None,
) -> None:
    """Evaluate an existing model against a dataset.

    Works on models michi never saw trained: an sklearn pickle, or anything
    exposing predict(X) via 'mymodule:my_model'. Reports metrics with
    confidence intervals, always compares against trivial baselines, scores
    each subgroup separately, and flags the mistakes that make good numbers
    misleading.
    """
    console = Console()
    try:
        from michi.adapters import load_model

        loaded = load_model(model)
        table = load_table(data, sample_rows=sample, full=full, seed=seed)
        manifest = evaluate_model(
            loaded,
            table,
            target=target,
            task=task,
            features=_split(features),
            slice_columns=_split(slice_by),
            bootstrap=bootstrap,
            seed=seed,
        )
    except MichiError as err:
        Console(stderr=True).print(f"[bold red]error[/] {err}")
        raise typer.Exit(code=2) from err

    render_evaluation(manifest, console, explain=explain)

    written: list[Path] = []
    if not no_save:
        destination = runs_dir / f"{manifest.run_id}.json"
        _write_manifest(manifest, destination)
        written.append(destination)
    if json_out is not None:
        _write_manifest(manifest, json_out)
        written.append(json_out)
    for destination in written:
        console.print(f"  [dim]wrote[/] {destination}")
    if written:
        console.print()

    if fail_under is not None:
        raise typer.Exit(code=_gate(manifest, fail_under, console))


def _split(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated option into a tuple of names."""
    if value is None:
        return None
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    return names or None


def _write_manifest(manifest: RunManifest, destination: Path) -> None:
    """Write a run manifest as formatted, UTF-8 JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _gate(manifest: RunManifest, expression: str, console: Console) -> int:
    """Return 0 when the named metric meets its threshold, 1 otherwise."""
    name, separator, raw_threshold = expression.partition("=")
    if not separator:
        msg = f"--fail-under expects 'metric=value' (got {expression!r})"
        raise typer.BadParameter(msg)
    try:
        threshold = float(raw_threshold)
    except ValueError as err:
        msg = f"--fail-under threshold must be a number (got {raw_threshold!r})"
        raise typer.BadParameter(msg) from err

    try:
        metric = manifest.metric(name.strip())
    except KeyError as err:
        available = ", ".join(item.name for item in manifest.metrics)
        msg = f"unknown metric {name.strip()!r}; this run recorded: {available}"
        raise typer.BadParameter(msg) from err

    passed = (
        metric.value >= threshold
        if metric.greater_is_better
        else metric.value <= threshold
    )
    direction = "at least" if metric.greater_is_better else "at most"
    if passed:
        console.print(
            f"  [green]gate passed[/] {metric.name} {metric.value:.4g} "
            f"({direction} {threshold:g})\n"
        )
        return 0
    console.print(
        f"  [bold red]gate failed[/] {metric.name} {metric.value:.4g} "
        f"(needed {direction} {threshold:g})\n"
    )
    return 1
