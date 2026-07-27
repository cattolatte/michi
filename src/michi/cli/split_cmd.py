"""``michi split`` — hold data out, in the way this data actually requires.

Design Principles
-----------------
- **The default is the safe one for the data's shape.** A classification
  target gets a stratified split, because an unstratified one on an imbalanced
  problem can hand a fold too few positives to score. Naming ``--group`` or
  ``--time`` overrides it, because those constraints are stronger than balance
  and michi cannot infer them.
- **Grouped and time splits exist because a random one lies.** Two rows from
  the same customer, or a test set drawn from before the training rows, both
  produce a score that will not survive contact with production. These are the
  two leaks a split can cause, and both need a column michi has no way to
  guess.
- **It writes files and a receipt.** The two datasets are the output; the
  printed summary states the strategy, the sizes, and — for a grouped split —
  that no group spans both sides, which is the property the whole thing exists
  to guarantee.
- **Never destructive.** The input is read and never written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from michi.cli.context import resolve_defaults
from michi.cli.errors import fail
from michi.core.errors import DataError, MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table

__all__ = ["split_command"]

STRATEGIES: tuple[str, ...] = ("auto", "random", "stratified", "group", "time")


def split_command(
    data: Annotated[
        Path | None,
        typer.Argument(help="Dataset to split. Falls back to `data` in michi.toml."),
    ] = None,
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Label column.")
    ] = None,
    test_size: Annotated[
        float, typer.Option("--test-size", help="Fraction held out, 0–1.")
    ] = 0.2,
    strategy: Annotated[
        str,
        typer.Option("--strategy", help="auto, random, stratified, group, or time."),
    ] = "auto",
    group: Annotated[
        str | None,
        typer.Option("--group", help="Keep rows sharing this column on one side."),
    ] = None,
    time_column: Annotated[
        str | None,
        typer.Option("--time", help="Split chronologically on this column."),
    ] = None,
    train_out: Annotated[
        Path, typer.Option("--train", help="Where to write the training rows.")
    ] = Path("train.csv"),
    test_out: Annotated[
        Path, typer.Option("--test", help="Where to write the held-out rows.")
    ] = Path("test.csv"),
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
) -> None:
    """Split a dataset into training and held-out files.

    Stratifies by default when the target is categorical. Pass --group to keep
    rows sharing an entity together, or --time to hold out the future — the
    two cases where a random split reports a score that will not survive
    contact with production.
    """
    console = Console()
    defaults = resolve_defaults()
    resolved_seed = defaults.number("seed", seed) or 0

    try:
        if not 0.0 < test_size < 1.0:
            msg = f"--test-size must be between 0 and 1 (got {test_size})"
            raise DataError(msg)
        if strategy not in STRATEGIES:
            known = ", ".join(STRATEGIES)
            msg = f"unknown strategy {strategy!r}; michi offers: {known}"
            raise DataError(msg)

        resolved_data = defaults.required_data(data)
        table = load_table(
            resolved_data, sample_rows=sample, full=full, seed=resolved_seed
        )
        frame = table.frame
        resolved_target, note = defaults.target_for(target, frame.columns)
        if note:
            console.print(f"  [dim]{note}[/]")

        chosen = _resolve_strategy(strategy, frame, resolved_target, group, time_column)
        train, test = _split(
            frame,
            strategy=chosen,
            target=resolved_target,
            group=group,
            time_column=time_column,
            test_size=test_size,
            seed=resolved_seed,
        )
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    _write(train, train_out)
    _write(test, test_out)
    _render(
        console,
        chosen=chosen,
        train=train,
        test=test,
        target=resolved_target,
        group=group,
        time_column=time_column,
        train_out=train_out,
        test_out=test_out,
        seed=resolved_seed,
    )


def _resolve_strategy(
    strategy: str,
    frame: Any,
    target: str | None,
    group: str | None,
    time_column: str | None,
) -> str:
    """Pick the split this data's shape calls for, when not told."""
    if strategy != "auto":
        return strategy
    # An explicit column is a constraint michi cannot infer and the user would
    # not have named by accident, so it outranks the balance heuristic.
    if group:
        return "group"
    if time_column:
        return "time"
    if target and target in frame.columns and _is_categorical(frame[target]):
        return "stratified"
    return "random"


def _is_categorical(column: Any) -> bool:
    """Whether a target looks like classes rather than a quantity."""
    from michi.evaluation.metrics import detect_task

    return bool(detect_task(column.to_numpy()) == "classification")


def _split(
    frame: Any,
    *,
    strategy: str,
    target: str | None,
    group: str | None,
    time_column: str | None,
    test_size: float,
    seed: int,
) -> tuple[Any, Any]:
    """Produce the two frames."""
    from sklearn.model_selection import (
        GroupShuffleSplit,
        StratifiedShuffleSplit,
        train_test_split,
    )

    if strategy == "time":
        if not time_column or time_column not in frame.columns:
            msg = "--time needs a column present in the data"
            raise DataError(msg)
        import pandas as pd

        stamps = pd.to_datetime(frame[time_column], errors="coerce", format="mixed")
        if stamps.isna().all():
            msg = f"could not read {time_column!r} as dates"
            raise DataError(msg)
        # Sorting rather than shuffling is the point: the held-out rows must be
        # the later ones, or the model is being asked to predict the past.
        ordered = frame.assign(_when=stamps).sort_values("_when").drop(columns="_when")
        cut = int(len(ordered) * (1 - test_size))
        return ordered.iloc[:cut], ordered.iloc[cut:]

    if strategy == "group":
        if not group or group not in frame.columns:
            msg = "--group needs a column present in the data"
            raise DataError(msg)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_index, test_index = next(
            splitter.split(frame, groups=frame[group].astype("object"))
        )
        return frame.iloc[train_index], frame.iloc[test_index]

    if strategy == "stratified":
        if not target or target not in frame.columns:
            msg = "a stratified split needs --target"
            raise DataError(msg)
        labels = frame[target]
        smallest = labels.value_counts().min()
        if smallest < 2:
            msg = (
                f"cannot stratify: the rarest class of {target!r} has "
                f"{smallest} row(s). Use --strategy random."
            )
            raise DataError(msg)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_index, test_index = next(splitter.split(frame, labels))
        return frame.iloc[train_index], frame.iloc[test_index]

    train, test = train_test_split(frame, test_size=test_size, random_state=seed)
    return train, test


def _render(
    console: Console,
    *,
    chosen: str,
    train: Any,
    test: Any,
    target: str | None,
    group: str | None,
    time_column: str | None,
    train_out: Path,
    test_out: Path,
    seed: int,
) -> None:
    """Print what was done, and the property it guarantees."""
    header = Text()
    header.append(" 道 ", style="bold red")
    header.append(" michi split", style="bold")
    header.append(f"  ·  {chosen}", style="bold cyan")
    console.print()
    console.print(header)
    console.print()

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("", style="dim")
    table.add_column("rows", justify="right")
    table.add_column("file", style="cyan")
    table.add_row("train", f"{len(train):,}", str(train_out))
    table.add_row("held out", f"{len(test):,}", str(test_out))
    console.print(Padding(table, (0, 0, 1, 2)))

    console.print(Padding(_guarantee(chosen, train, test, target, group), (0, 0, 1, 2)))
    console.print(
        Padding(
            Text(
                f"seed {seed}  ·  the input file was not modified\n"
                f"Next:  michi bench {train_out} --target {target or '<col>'}",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )


def _guarantee(
    chosen: str, train: Any, test: Any, target: str | None, group: str | None
) -> Text:
    """State the property this split was chosen to provide, and verify it."""
    text = Text()
    if chosen == "group" and group:
        overlap = set(train[group]) & set(test[group])
        if overlap:
            # Reported rather than raised: the split is still usable, and a
            # silent guarantee that did not hold is worse than a loud one.
            text.append("warning: ", style="bold yellow")
            text.append(f"{len(overlap)} value(s) of {group!r} appear on both sides.")
            return text
        text.append("No value of ", style="dim")
        text.append(group, style="bold")
        text.append(
            " appears on both sides — the leak a random split would have "
            "caused is absent.",
            style="dim",
        )
        return text
    if chosen == "time":
        text.append(
            "The held-out rows are the later ones, so the model is never "
            "asked to predict the past.",
            style="dim",
        )
        return text
    if chosen == "stratified" and target:
        text.append("Class balance preserved on both sides of ", style="dim")
        text.append(target, style="bold")
        text.append(".", style="dim")
        return text
    text.append(
        "Rows assigned at random. If rows share an entity, or the data is a "
        "time series,\nthis score will be optimistic — see --group and --time.",
        style="dim",
    )
    return text


def _write(frame: Any, destination: Path) -> None:
    """Write a split in the format its extension implies."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    elif suffix == ".tsv":
        frame.to_csv(destination, index=False, sep="\t", encoding="utf-8")
    else:
        frame.to_csv(destination, index=False, encoding="utf-8")
