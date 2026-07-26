"""The ``michi clean``, ``apply``, and ``export`` commands.

Design Principles
-----------------
- ``clean`` never touches the input data. It writes a recipe; ``apply``
  produces data, always to a new file.
- The wizard prints the equivalent non-interactive command when it finishes,
  so an exploratory session always converts into something scriptable.
- Flags cover everything the wizard can do, so michi stays usable from a
  script, from CI, and by anyone automating it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from michi.core.errors import MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table

__all__ = ["apply_command", "clean_command", "export_command"]


def clean_command(
    data: Annotated[
        Path, typer.Argument(help="Dataset to author a recipe for.", show_default=False)
    ],
    output: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the recipe.")
    ] = Path("michi.recipe.yaml"),
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Label column.")
    ] = None,
    drop: Annotated[
        str | None, typer.Option("--drop", help="Comma-separated columns to drop.")
    ] = None,
    dedupe: Annotated[
        bool, typer.Option("--dedupe", help="Remove exactly duplicated rows.")
    ] = False,
    cast: Annotated[
        list[str] | None,
        typer.Option("--cast", help="COLUMN=numeric|datetime|category|string."),
    ] = None,
    impute: Annotated[
        list[str] | None,
        typer.Option(
            "--impute",
            help="COLUMN=median|mean|most_frequent|constant|drop_rows.",
        ),
    ] = None,
    clip: Annotated[
        str | None,
        typer.Option("--clip", help="Comma-separated columns to clip to 1–99%."),
    ] = None,
    encode: Annotated[
        list[str] | None,
        typer.Option("--encode", help="COLUMN=onehot|ordinal."),
    ] = None,
    scale: Annotated[
        list[str] | None,
        typer.Option("--scale", help="COLUMN=standard|minmax|robust."),
    ] = None,
    no_input: Annotated[
        bool,
        typer.Option("--no-input", help="Never prompt; use only the flags given."),
    ] = False,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Seed for sampling.")] = 0,
) -> None:
    """Author a cleaning recipe — interactively, or entirely from flags.

    Profiles the data, walks the issues it found, and records your decisions
    as a recipe file. Your data is never modified: 'michi apply' produces the
    cleaned copy, and 'michi export' compiles the recipe into pipeline code.
    """
    console = Console()
    try:
        from michi.inspection import profile_table
        from michi.recipes import (
            recipe_from_flags,
            write_recipe,
        )

        table = load_table(data, sample_rows=sample, full=full, seed=seed)
        profile = profile_table(table, target=target)

        flags_given = any([drop, dedupe, cast, impute, clip, encode, scale])
        if flags_given or no_input:
            recipe = recipe_from_flags(
                profile,
                drop=_split(drop),
                dedupe=dedupe,
                cast=_pairs(cast, "cast"),
                impute=_pairs(impute, "impute"),
                clip=_split(clip),
                encode=_pairs(encode, "encode"),
                scale=_pairs(scale, "scale"),
                target=target,
            )
        else:
            recipe = _run_wizard(profile, console, target=target)
    except MichiError as err:
        Console(stderr=True).print(f"[bold red]error[/] {err}")
        raise typer.Exit(code=2) from err

    if not recipe.steps:
        console.print("\n  No steps chosen — nothing to write.\n")
        raise typer.Exit()

    write_recipe(recipe, output)
    console.print()
    console.print(Padding(_summary(recipe, output), (0, 0, 1, 2)))
    console.print(Padding(_reproduce(recipe, str(data)), (0, 0, 1, 2)))
    console.print(
        Padding(
            Text(
                f"Next:  michi apply {output} {data} -o clean.parquet\n"
                f"       michi export {output} -o pipeline.py",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )


def apply_command(
    recipe_path: Annotated[
        Path, typer.Argument(help="Recipe to execute.", show_default=False)
    ],
    data: Annotated[
        Path, typer.Argument(help="Dataset to transform.", show_default=False)
    ],
    output: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Where to write the transformed data."),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Fail when the data lacks columns the recipe names.",
        ),
    ] = True,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Seed for sampling.")] = 0,
) -> None:
    """Apply a recipe to a dataset, writing a new file.

    The input is never modified. Steps that learn from data — imputation,
    encoding, scaling — are fitted on the rows given here, which michi reports
    so the leakage risk is never hidden.
    """
    console = Console()
    try:
        from michi.recipes import apply_recipe, load_recipe

        recipe = load_recipe(recipe_path)
        table = load_table(data, sample_rows=sample, full=full, seed=seed)
        result = apply_recipe(recipe, table.frame, strict=strict)
    except MichiError as err:
        Console(stderr=True).print(f"[bold red]error[/] {err}")
        raise typer.Exit(code=2) from err

    console.print()
    shape = Text()
    shape.append(f"{result.rows_before:,} × {result.columns_before}", style="dim")
    shape.append("  →  ")
    shape.append(f"{result.rows_after:,} × {result.columns_after}", style="bold")
    console.print(Padding(shape, (0, 0, 1, 2)))

    for note in result.notes:
        console.print(Padding(Text(f"note: {note}", style="yellow"), (0, 0, 0, 2)))
    if result.notes:
        console.print()

    if output is None:
        console.print(
            Padding(
                Text("Pass --out to write the result to a file.", style="dim"),
                (0, 0, 1, 2),
            )
        )
        return

    try:
        _write_frame(result.frame, output)
    except MichiError as err:
        Console(stderr=True).print(f"[bold red]error[/] {err}")
        raise typer.Exit(code=2) from err
    console.print(Padding(Text(f"wrote {output}", style="dim"), (0, 0, 1, 2)))


def export_command(
    recipe_path: Annotated[
        Path, typer.Argument(help="Recipe to compile.", show_default=False)
    ],
    output: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Where to write the generated module."),
    ] = None,
) -> None:
    """Compile a recipe into readable Python you own.

    The generated module imports pandas and scikit-learn — never michi.
    Deterministic steps become a prepare() function; steps that learn from
    data become an sklearn pipeline you fit inside cross-validation.
    """
    console = Console()
    try:
        from michi.recipes import export_recipe, load_recipe

        recipe = load_recipe(recipe_path)
        code = export_recipe(recipe, module_name=output.stem if output else "pipeline")
    except MichiError as err:
        Console(stderr=True).print(f"[bold red]error[/] {err}")
        raise typer.Exit(code=2) from err

    if output is None:
        console.print(code, markup=False, highlight=False)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(code, encoding="utf-8")
    console.print(f"\n  [dim]wrote[/] {output}")
    console.print(
        "  [dim]it imports pandas and scikit-learn only — michi is not a "
        "dependency of this file[/]\n"
    )


# --- the wizard ------------------------------------------------------------


def _run_wizard(profile, console: Console, *, target: str | None):  # type: ignore[no-untyped-def]
    """Walk the findings, one question per group, and build a recipe."""
    from michi.recipes import questions_for, recipe_from_answers

    questions = questions_for(profile)
    console.print()
    console.print(Padding(_wizard_header(profile, questions), (0, 0, 1, 2)))

    if not questions:
        console.print(
            Padding(
                Text("Nothing stood out — no decisions to make.", style="green"),
                (0, 0, 1, 2),
            )
        )
        return recipe_from_answers(profile, [], target=target)

    answers: list[tuple[object, str]] = []
    for index, question in enumerate(questions, start=1):
        answers.append((question, _ask(question, index, len(questions), console)))
    return recipe_from_answers(profile, answers, target=target)  # type: ignore[arg-type]


def _ask(question, index: int, total: int, console: Console) -> str:  # type: ignore[no-untyped-def]
    """Present one grouped decision and return the chosen key."""
    import questionary

    console.print()
    heading = Text()
    heading.append(f"  [{index}/{total}]  ", style="dim")
    heading.append(question.prompt, style="bold")
    console.print(heading)
    if question.detail:
        console.print(Padding(Text(question.detail, style="dim"), (0, 0, 0, 8)))
    if question.is_bulk:
        shown = ", ".join(question.columns[:6])
        extra = len(question.columns) - 6
        more = f" (+{extra} more)" if extra > 0 else ""
        console.print(Padding(Text(f"{shown}{more}", style="cyan"), (0, 0, 0, 8)))

    answer = questionary.select(
        "  What would you like to do?",
        choices=[
            questionary.Choice(title=choice.label, value=choice.key)
            for choice in question.choices
        ],
        default=question.choices[-1].key,
        instruction=" ",
    ).ask()
    return str(answer) if answer is not None else "keep"


def _wizard_header(profile, questions) -> Text:  # type: ignore[no-untyped-def]
    text = Text()
    text.append(" 道 ", style="bold red")
    text.append(" michi clean", style="bold")
    text.append("  ·  ", style="dim")
    text.append(f"{len(questions)} decision(s)", style="bold cyan")
    text.append("\n\n")
    text.append(
        f"{profile.n_rows:,} rows × {profile.n_columns} columns. "
        "michi lists the options; you choose. Your data is not modified — "
        "this writes a recipe.",
        style="dim",
    )
    return text


def _summary(recipe, output: Path) -> Text:  # type: ignore[no-untyped-def]
    text = Text()
    text.append(f"wrote {output}", style="bold")
    text.append(f"  ·  {len(recipe.steps)} step(s)\n", style="dim")
    for step in recipe.steps:
        text.append("  · ", style="dim")
        text.append(step.op, style="cyan")
        if step.columns:
            shown = ", ".join(step.columns[:5])
            extra = f" +{len(step.columns) - 5}" if len(step.columns) > 5 else ""
            text.append(f"  {shown}{extra}")
        text.append("\n")
    return text


def _reproduce(recipe, data_path: str) -> Text:  # type: ignore[no-untyped-def]
    from michi.recipes import command_for

    text = Text()
    text.append("To reproduce this without the prompts:\n", style="dim")
    text.append(f"  {command_for(recipe, data_path)}", style="cyan")
    return text


# --- helpers ---------------------------------------------------------------


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _pairs(values: list[str] | None, flag: str) -> tuple[tuple[str, str], ...]:
    """Parse repeated ``COLUMN=value`` options."""
    if not values:
        return ()
    parsed: list[tuple[str, str]] = []
    for item in values:
        column, separator, setting = item.partition("=")
        if not separator or not column.strip() or not setting.strip():
            msg = f"--{flag} expects COLUMN=value (got {item!r})"
            raise typer.BadParameter(msg)
        parsed.append((column.strip(), setting.strip()))
    return tuple(parsed)


def _write_frame(frame, destination: Path) -> None:  # type: ignore[no-untyped-def]
    """Write a transformed frame in the format its extension implies."""
    from michi.core.errors import DataError

    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    elif suffix in {".csv", ".txt"}:
        frame.to_csv(destination, index=False, encoding="utf-8")
    elif suffix == ".tsv":
        frame.to_csv(destination, index=False, sep="\t", encoding="utf-8")
    else:
        msg = (
            f"cannot write {suffix or destination.name!r}; use .csv, .tsv, or .parquet"
        )
        raise DataError(msg)
