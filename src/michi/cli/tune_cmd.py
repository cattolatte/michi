"""The ``michi tune`` command — hyperparameter search, honestly scored.

Design Principles
-----------------
- **The space is printable.** ``--list-space`` shows exactly what will be
  searched before anything runs, and ``--space file.yaml`` replaces it. A
  search space you cannot inspect is a modelling decision michi made for you.
- **Nested, always.** The reported score comes from folds the search never
  saw. The optimistic inner score is printed *next to* it rather than instead
  of it, because the gap between them is the lesson.
- **Tuning is not a verdict.** michi reports what the search found and how
  much of it survived; it never says to ship the tuned model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from michi.cli.context import resolve_defaults
from michi.cli.errors import fail
from michi.core.errors import MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table

__all__ = ["tune_command"]


def tune_command(
    data: Annotated[
        Path | None,
        typer.Argument(help="Dataset. Falls back to `data` in michi.toml."),
    ] = None,
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Label column.")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to tune.")
    ] = "hist-gbm",
    strategy: Annotated[
        str,
        typer.Option("--strategy", help="random, halving, or grid."),
    ] = "random",
    candidates: Annotated[
        int, typer.Option("--candidates", help="Configurations to try (random only).")
    ] = 30,
    space: Annotated[
        Path | None,
        typer.Option("--space", help="YAML search space, replacing the built-in one."),
    ] = None,
    list_space: Annotated[
        bool,
        typer.Option("--list-space", help="Print the search space and exit."),
    ] = False,
    save_params: Annotated[
        Path | None,
        typer.Option("--save-params", help="Write the winning parameters as YAML."),
    ] = None,
    recipe: Annotated[
        Path | None, typer.Option("--recipe", help="Cleaning recipe to apply first.")
    ] = None,
    cv: Annotated[int, typer.Option("--cv", help="Outer cross-validation folds.")] = 5,
    inner_cv: Annotated[
        int, typer.Option("--inner-cv", help="Folds used inside the search.")
    ] = 3,
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric",
            help="Rank and test by this metric instead of michi's default.",
        ),
    ] = None,
    group: Annotated[
        str | None,
        typer.Option(
            "--group",
            help="Keep rows sharing this column in one fold. Use it whenever "
            "rows share an entity.",
        ),
    ] = None,
    task: Annotated[
        str | None,
        typer.Option("--task", help="Force `classification` or `regression`."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
) -> None:
    """Search hyperparameters, and score the winner on untouched folds.

    The search runs inside each training fold, so the score reported comes
    from data no configuration was chosen on. michi prints the optimistic
    inner score alongside it, because the gap between them is the point.
    """
    console = Console()
    defaults = resolve_defaults()
    resolved_seed = defaults.number("seed", seed) or 0

    try:
        from michi.bench.tuning import load_space, search_space, tune_model

        grid = load_space(space) if space is not None else search_space(model)
        if list_space:
            _print_space(console, model, grid, source=space)
            return

        resolved_data = defaults.required_data(data)
        table = load_table(
            resolved_data, sample_rows=sample, full=full, seed=resolved_seed
        )
        resolved_target, note = defaults.target_for(target, table.frame.columns)
        if note:
            console.print(f"  [dim]{note}[/]")
        if not resolved_target or resolved_target not in table.frame.columns:
            msg = (
                "tuning needs a target column: pass --target, "
                "or set `target` in michi.toml"
            )
            raise MichiError(msg)

        from michi.evaluation.metrics import detect_task

        labels = table.frame[resolved_target]
        group_values = (
            table.frame[group].astype("object").to_numpy()
            if group and group in table.frame.columns
            else None
        )
        # The grouping column identifies the entity; leaving it as a feature
        # would hand the model the very thing grouping exists to hide.
        dropped = [resolved_target] + ([group] if group_values is not None else [])
        features = table.frame.drop(columns=dropped)
        loaded_recipe = None
        if recipe is not None:
            from michi.recipes import apply_deterministic, load_recipe

            loaded_recipe = load_recipe(recipe)
            features = apply_deterministic(loaded_recipe, features)

        console.print()
        console.print(_header(model, strategy))
        console.print()
        console.print(
            Padding(
                Text(
                    f"searching {_size(grid)} configurations  ·  "
                    f"{cv}-fold outer  ·  {inner_cv}-fold inner  ·  "
                    f"seed {resolved_seed}",
                    style="dim",
                ),
                (0, 0, 1, 2),
            )
        )

        result = tune_model(
            features,
            labels.to_numpy(),
            model=model,
            task=task or detect_task(labels.to_numpy()),
            space=grid,
            strategy=strategy,
            candidates=candidates,
            folds=cv,
            inner_folds=inner_cv,
            seed=resolved_seed,
            recipe=loaded_recipe,
            groups=group_values,
            metric=metric,
        )
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    _render(console, result)

    if save_params is not None:
        import yaml

        save_params.parent.mkdir(parents=True, exist_ok=True)
        save_params.write_text(
            yaml.safe_dump(result.best_params, sort_keys=True), encoding="utf-8"
        )
        console.print(
            Padding(
                Text(
                    f"wrote {save_params}  —  "
                    f"michi fit --model {result.model} --params {save_params}",
                    style="dim",
                ),
                (0, 0, 1, 2),
            )
        )


def _header(model: str, strategy: str) -> Text:
    text = Text()
    text.append(" 道 ", style="bold red")
    text.append(" michi tune", style="bold")
    text.append("  ·  ", style="dim")
    text.append(model, style="bold cyan")
    text.append(f"  ·  {strategy} search", style="dim")
    return text


def _size(space: dict[str, list[object]]) -> int:
    total = 1
    for values in space.values():
        total *= max(len(values), 1)
    return total


def _print_space(
    console: Console, model: str, space: dict[str, list[object]], *, source: Path | None
) -> None:
    """Show what would be searched, so nothing is decided out of sight."""
    console.print()
    console.print(_header(model, "space"))
    console.print()
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("parameter", style="cyan", no_wrap=True)
    table.add_column("values")
    for name, values in sorted(space.items()):
        table.add_row(
            name.removeprefix("model__"), ", ".join(repr(item) for item in values)
        )
    console.print(Padding(table, (0, 0, 1, 2)))
    origin = f"from {source}" if source else "michi's built-in space"
    console.print(
        Padding(
            Text(
                f"{_size(space)} combinations  ·  {origin}\n"
                "Replace it with --space my_space.yaml; the file maps a "
                "parameter name to a list of values.",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )


def _render(console: Console, result: object) -> None:
    """Print what the search found, honest number first."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("", style="dim", no_wrap=True)
    table.add_column(result.metric, justify="right")  # type: ignore[attr-defined]
    table.add_row("tuned (held-out folds)", f"{result.outer_score:.4g}")  # type: ignore[attr-defined]
    table.add_row("defaults (same folds)", f"{result.baseline_score:.4g}")  # type: ignore[attr-defined]
    table.add_row("search's own best", f"{result.inner_score:.4g}")  # type: ignore[attr-defined]
    console.print(Padding(table, (0, 0, 1, 2)))

    chosen = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    chosen.add_column("parameter", style="cyan", no_wrap=True)
    chosen.add_column("chosen")
    for name, value in sorted(result.best_params.items()):  # type: ignore[attr-defined]
        chosen.add_row(name, repr(value))
    console.print(Padding(chosen, (0, 0, 1, 2)))

    console.print(Padding(_verdict(result), (0, 0, 1, 2)))


def _verdict(result: object) -> Text:
    """State what the tuning bought, and what the inner score was worth."""
    improvement = result.improvement  # type: ignore[attr-defined]
    optimism = result.optimism  # type: ignore[attr-defined]
    metric = result.metric  # type: ignore[attr-defined]

    text = Text()
    text.append("Verdict  ", style="bold")
    if improvement > 0:
        text.append(
            f"tuning gained {improvement:.4g} of {metric} over the defaults, "
            f"measured on folds the search never saw."
        )
    elif improvement == 0:
        text.append(
            f"tuning matched the defaults exactly on {metric}. The search "
            "found nothing the default configuration did not already have."
        )
    else:
        text.append(
            f"tuning did {abs(improvement):.4g} of {metric} *worse* than the "
            "defaults on held-out folds. That happens: a search that picks on "
            "one split can pick something that does not carry to another."
        )
    text.append("\n")
    text.append(
        f"The search's own best score was {optimism:.4g} better than the "
        "held-out result.\nThat gap is what reporting an inner score as "
        "performance would have hidden.",
        style="dim",
    )
    return text
