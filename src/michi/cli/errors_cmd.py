"""``michi errors`` — what did it get wrong, and do the mistakes rhyme?

Design Principles
-----------------
- **The next question after a score.** ``eval`` says 0.88. Every practitioner
  then wants the rows in the 0.12, and writes the same pandas to get them.
- **Confident and wrong is the interesting case.** A mistake the model was
  unsure about is the model behaving correctly at the edge. A mistake it was
  certain of is a mislabelled row, a leak, or a region the features do not
  describe — so the listing is ordered by confidence, not by index.
- **Patterns are described, not diagnosed.** michi reports that errors
  concentrate in a feature range or a category. Why they do is domain
  knowledge michi does not have.
- **The rows are a file.** ``--out`` writes them, because the useful next step
  is opening them in something that is not a terminal.
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

__all__ = ["errors_command"]

_MIN_GROUP = 10
"""Rows a subgroup needs before its error rate means anything."""

_LIFT = 1.5
"""How much worse than average a group must be before michi mentions it."""


def errors_command(
    model: Annotated[
        str,
        typer.Argument(
            help="Model: a pickle/joblib file, or 'mymodule:my_model'.",
            show_default=False,
        ),
    ],
    data: Annotated[
        Path | None,
        typer.Argument(help="Labelled data. Falls back to `data` in michi.toml."),
    ] = None,
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Label column.")
    ] = None,
    show: Annotated[
        int, typer.Option("--show", help="How many mistakes to list.")
    ] = 10,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write every mistake here.")
    ] = None,
    recipe: Annotated[
        Path | None, typer.Option("--recipe", help="Cleaning recipe to apply first.")
    ] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
) -> None:
    """Show the rows a model got wrong, worst first, and what they share.

    Orders mistakes by how confident the model was, because a confident error
    is a mislabelled row, a leak, or a blind spot — while an unsure one is the
    model behaving correctly at the edge.
    """
    console = Console()
    defaults = resolve_defaults()

    try:
        frame, wrong, confidence = _find(
            model=model,
            data=defaults.required_data(data),
            target=defaults.text("target", target),
            recipe=defaults.path("recipe", recipe),
            sample=sample,
            full=full,
        )
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    _render(console, frame, wrong, confidence, show=show)

    if out is not None and len(wrong):
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in {".parquet", ".pq"}:
            wrong.to_parquet(out, index=False)
        else:
            wrong.to_csv(out, index=False, encoding="utf-8")
        console.print(f"  [dim]wrote {len(wrong):,} mistake(s) to {out}[/]\n")


def _find(
    *,
    model: str,
    data: Path,
    target: str | None,
    recipe: Path | None,
    sample: int,
    full: bool,
) -> tuple[Any, Any, bool]:
    """Predict, keep the rows that were wrong, and rank them by confidence."""
    import numpy as np

    from michi.adapters import load_model

    if not target:
        msg = "error analysis needs a target: pass --target, or set it in michi.toml"
        raise DataError(msg)

    table = load_table(data, sample_rows=sample, full=full, seed=0)
    frame = table.frame
    if recipe is not None:
        from michi.recipes import apply_deterministic, load_recipe

        frame = apply_deterministic(load_recipe(recipe), frame)
    if target not in frame.columns:
        msg = f"target {target!r} is not a column of {data.name}"
        raise DataError(msg)

    truth = frame[target]
    features = frame.drop(columns=[target])
    loaded = load_model(model)
    predictions = np.asarray(loaded.predict(features)).reshape(len(frame), -1)[:, 0]

    numeric_target = truth.dtype.kind in "if"
    if numeric_target and len(set(truth.to_numpy().tolist())) > 20:
        # A regression has no "wrong", only "far". The residual is the
        # equivalent ordering, and the largest ones are the interesting rows.
        residual = np.abs(truth.to_numpy(dtype=float) - predictions.astype(float))
        annotated = frame.assign(predicted=predictions, error=residual)
        wrong = annotated.sort_values("error", ascending=False)
        return annotated, wrong, False

    mistaken = predictions != truth.to_numpy()
    annotated = frame.assign(predicted=predictions, correct=~mistaken)
    wrong = annotated[mistaken].copy()

    probabilities = loaded.predict_proba(features)
    if probabilities is not None and len(wrong):
        # How sure the model was of the answer it gave — the number that makes
        # a confident mistake findable.
        certainty = np.asarray(probabilities).max(axis=1)
        wrong["confidence"] = certainty[mistaken]
        wrong = wrong.sort_values("confidence", ascending=False)
        return annotated, wrong, True
    return annotated, wrong, False


def _render(
    console: Console, frame: Any, wrong: Any, confidence: bool, *, show: int
) -> None:
    """Print the count, the worst rows, and any pattern in them."""
    header = Text()
    header.append(" 道 ", style="bold red")
    header.append(" michi errors", style="bold")
    console.print()
    console.print(header)
    console.print()

    total = len(frame)
    if not len(wrong):
        console.print(
            Padding(Text("No mistakes on this data.", style="green"), (0, 0, 1, 2))
        )
        return

    rate = len(wrong) / total if total else 0.0
    summary = Text(style="dim")
    summary.append(
        f"{len(wrong):,} of {total:,} rows wrong ({rate:.1%})"
        + ("  ·  ordered by how sure the model was" if confidence else "")
    )
    console.print(Padding(summary, (0, 0, 1, 2)))

    console.print(Padding(_rows_table(wrong, show, confidence), (0, 0, 1, 2)))

    patterns = _patterns(frame, wrong)
    if patterns:
        console.print(Padding(patterns, (0, 0, 1, 2)))

    if confidence:
        console.print(
            Padding(
                Text(
                    "A confident mistake is a mislabelled row, a leak, or a "
                    "region the features\ndo not describe. An unsure one is "
                    "the model behaving correctly at the edge.",
                    style="dim",
                ),
                (0, 0, 1, 2),
            )
        )


def _rows_table(wrong: Any, show: int, confidence: bool) -> Table:
    """The worst mistakes, with the columns most likely to explain them."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    interesting = [
        name
        for name in wrong.columns
        if name not in {"correct", "confidence", "predicted", "error"}
    ][:5]

    if confidence:
        table.add_column("sure", justify="right", style="bold red")
    table.add_column("predicted", style="cyan")
    for name in interesting:
        table.add_column(str(name), overflow="fold")

    for _, row in wrong.head(max(show, 1)).iterrows():
        cells: list[str] = []
        if confidence:
            cells.append(f"{row['confidence']:.0%}")
        cells.append(str(row["predicted"]))
        cells.extend(_cell(row[name]) for name in interesting)
        table.add_row(*cells)
    return table


def _cell(value: Any) -> str:
    """Render one value compactly, whatever it is."""
    if value is None or value != value:  # NaN
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)[:18]


def _patterns(frame: Any, wrong: Any) -> Any:
    """Subgroups where the model fails far more than it does overall.

    Described, never diagnosed: michi can see that errors concentrate in a
    category, and cannot see why.
    """
    baseline = len(wrong) / len(frame) if len(frame) else 0.0
    if not baseline:
        return None

    lines: list[str] = []
    for name in frame.columns:
        if name in {"correct", "confidence", "predicted", "error"}:
            continue
        column = frame[name]
        if column.nunique(dropna=True) > 12 or column.nunique(dropna=True) < 2:
            continue
        for value, group in frame.groupby(column, observed=True):
            if len(group) < _MIN_GROUP:
                continue
            hits = int((~group["correct"]).sum()) if "correct" in group else 0
            rate = hits / len(group)
            if rate >= baseline * _LIFT and hits:
                lines.append(
                    f"{name} = {value!r}: {rate:.1%} wrong "
                    f"({hits} of {len(group)}), against {baseline:.1%} overall"
                )
    if not lines:
        return None

    text = Text()
    text.append("Where they concentrate\n\n", style="bold")
    for line in lines[:8]:
        text.append(f"  · {line}\n", style="none")
    return text
