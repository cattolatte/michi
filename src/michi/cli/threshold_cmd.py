"""``michi threshold`` — the decision cutoff nobody chose.

Design Principles
-----------------
- **0.5 is a default, not a decision.** A classifier outputs a probability;
  turning it into a label needs a cutoff, and almost every tool picks 0.5
  silently. On imbalanced data, or when a false negative costs more than a
  false positive, that is the wrong number and nothing says so.
- **The whole curve, and the user picks.** michi prints what every cutoff buys
  and costs. It marks the cutoff that maximises a stated objective *because
  the user stated it*, and never says which objective to want.
- **Costs are the user's, and optional.** ``--cost fn=10,fp=1`` says a missed
  positive is ten times worse. michi has no idea whether that is true, so it
  has no default beyond treating them equally.
"""

from __future__ import annotations

from dataclasses import dataclass
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

__all__ = ["threshold_command"]

OBJECTIVES: tuple[str, ...] = ("f1", "balanced_accuracy", "precision", "recall", "cost")


@dataclass(frozen=True, slots=True)
class _Row:
    """One candidate cutoff and what it produces."""

    cutoff: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    predicted_positive: int
    false_negatives: int
    false_positives: int
    cost: float


def threshold_command(
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
    positive: Annotated[
        str | None,
        typer.Option("--positive", help="Which class counts as positive."),
    ] = None,
    objective: Annotated[
        str,
        typer.Option("--objective", help="What to mark: " + ", ".join(OBJECTIVES)),
    ] = "f1",
    cost: Annotated[
        str | None,
        typer.Option("--cost", help="Relative costs, e.g. 'fn=10,fp=1'."),
    ] = None,
    steps: Annotated[
        int, typer.Option("--steps", help="How many cutoffs to evaluate.")
    ] = 19,
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
    """Show what every decision cutoff buys, and what it costs.

    A classifier outputs a probability; 0.5 is a default nobody chose. This
    prints precision, recall, and the errors at each cutoff so you can pick
    one deliberately — michi marks the best for the objective you name, and
    has no opinion about which objective to want.
    """
    console = Console()
    defaults = resolve_defaults()

    try:
        if objective not in OBJECTIVES:
            known = ", ".join(OBJECTIVES)
            msg = f"unknown objective {objective!r}; michi offers: {known}"
            raise DataError(msg)
        costs = _costs(cost)
        if objective == "cost" and cost is None:
            msg = "--objective cost needs --cost, e.g. --cost fn=10,fp=1"
            raise DataError(msg)

        rows, positive_label, base_rate = _measure(
            model=model,
            data=defaults.required_data(data),
            target=defaults.text("target", target),
            positive=positive,
            recipe=defaults.path("recipe", recipe),
            steps=steps,
            costs=costs,
            sample=sample,
            full=full,
        )
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    _render(console, rows, objective, positive_label, base_rate, costs)


def _costs(spec: str | None) -> tuple[float, float]:
    """Parse ``fn=10,fp=1`` into (false-negative, false-positive) weights."""
    if not spec:
        return 1.0, 1.0
    weights = {"fn": 1.0, "fp": 1.0}
    for part in spec.split(","):
        key, separator, value = part.partition("=")
        if not separator or key.strip() not in weights:
            msg = f"--cost expects fn=… and/or fp=… (got {part!r})"
            raise DataError(msg)
        try:
            weights[key.strip()] = float(value)
        except ValueError as err:
            msg = f"--cost values must be numbers (got {value!r})"
            raise DataError(msg) from err
    return weights["fn"], weights["fp"]


def _measure(
    *,
    model: str,
    data: Path,
    target: str | None,
    positive: str | None,
    recipe: Path | None,
    steps: int,
    costs: tuple[float, float],
    sample: int,
    full: bool,
) -> tuple[list[_Row], Any, float]:
    """Score every cutoff against the labels."""
    import numpy as np

    from michi.adapters import load_model

    if not target:
        msg = (
            "a threshold needs a target column: pass --target, or set it in michi.toml"
        )
        raise DataError(msg)

    table = load_table(data, sample_rows=sample, full=full, seed=0)
    frame = table.frame
    if recipe is not None:
        from michi.recipes import apply_deterministic, load_recipe

        frame = apply_deterministic(load_recipe(recipe), frame)
    if target not in frame.columns:
        msg = f"target {target!r} is not a column of {data.name}"
        raise DataError(msg)

    labels = frame[target]
    features = frame.drop(columns=[target])
    loaded = load_model(model)
    probabilities = loaded.predict_proba(features)
    if probabilities is None:
        msg = (
            "this model does not produce probabilities, so it has no cutoff to "
            "choose. `michi eval` scores its labels directly."
        )
        raise DataError(msg)

    classes = list(loaded.classes or range(probabilities.shape[1]))
    if len(classes) != 2:
        msg = (
            f"a decision cutoff applies to two classes; this model has "
            f"{len(classes)}. Evaluate it with `michi eval` instead."
        )
        raise DataError(msg)

    chosen = _positive_class(classes, positive)
    column = classes.index(chosen)
    scores = np.asarray(probabilities)[:, column]
    truth = (labels.to_numpy() == chosen).astype(int)
    base_rate = float(truth.mean())

    fn_cost, fp_cost = costs
    rows: list[_Row] = []
    for cutoff in np.linspace(0.05, 0.95, max(steps, 2)):
        predicted = (scores >= cutoff).astype(int)
        true_positive = int(((predicted == 1) & (truth == 1)).sum())
        false_positive = int(((predicted == 1) & (truth == 0)).sum())
        false_negative = int(((predicted == 0) & (truth == 1)).sum())
        true_negative = int(((predicted == 0) & (truth == 0)).sum())

        precision = true_positive / (true_positive + false_positive or 1)
        recall = true_positive / (true_positive + false_negative or 1)
        specificity = true_negative / (true_negative + false_positive or 1)
        rows.append(
            _Row(
                cutoff=float(cutoff),
                precision=precision,
                recall=recall,
                f1=(
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
                balanced_accuracy=(recall + specificity) / 2,
                predicted_positive=int(predicted.sum()),
                false_negatives=false_negative,
                false_positives=false_positive,
                cost=fn_cost * false_negative + fp_cost * false_positive,
            )
        )
    return rows, chosen, base_rate


def _positive_class(classes: list[Any], requested: str | None) -> Any:
    """Decide which label counts as positive."""
    if requested is None:
        # The second class is sklearn's own convention for the positive column
        # of predict_proba, so following it makes michi agree with every other
        # tool a user will compare against.
        return classes[1]
    for item in classes:
        if str(item) == requested:
            return item
    known = ", ".join(str(item) for item in classes)
    msg = f"--positive {requested!r} is not a class of this model; it has: {known}"
    raise DataError(msg)


def _render(
    console: Console,
    rows: list[_Row],
    objective: str,
    positive: Any,
    base_rate: float,
    costs: tuple[float, float],
) -> None:
    """Print the curve, marking the cutoff the stated objective prefers."""
    header = Text()
    header.append(" 道 ", style="bold red")
    header.append(" michi threshold", style="bold")
    header.append(f"  ·  positive = {positive}", style="dim")
    console.print()
    console.print(header)
    console.print()
    console.print(
        Padding(
            Text(
                f"{base_rate:.1%} of rows are positive  ·  "
                f"marking the best {objective}",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )

    best = _best(rows, objective)
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("", width=1)
    table.add_column("cutoff", justify="right")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("flagged", justify="right")
    table.add_column("missed", justify="right", style="dim")
    table.add_column("false alarms", justify="right", style="dim")
    if costs != (1.0, 1.0):
        table.add_column("cost", justify="right")

    for row in rows:
        marked = row is best
        style = "bold green" if marked else ""
        cells = [
            Text("▸" if marked else "", style="green"),
            Text(f"{row.cutoff:.2f}", style=style),
            Text(f"{row.precision:.3f}", style=style),
            Text(f"{row.recall:.3f}", style=style),
            Text(f"{row.f1:.3f}", style=style),
            Text(f"{row.predicted_positive:,}", style=style),
            Text(f"{row.false_negatives:,}"),
            Text(f"{row.false_positives:,}"),
        ]
        if costs != (1.0, 1.0):
            cells.append(Text(f"{row.cost:,.0f}", style=style))
        table.add_row(*cells)
    console.print(Padding(table, (0, 0, 1, 2)))

    note = Text()
    note.append("Verdict  ", style="bold")
    note.append(
        f"the best {objective} on this data is at {best.cutoff:.2f}, not 0.50. "
        f"At that cutoff\nthe model flags {best.predicted_positive:,} rows, "
        f"misses {best.false_negatives:,}, and raises "
        f"{best.false_positives:,} false alarms."
    )
    console.print(Padding(note, (0, 0, 1, 2)))
    console.print(
        Padding(
            Text(
                "Which trade to take is yours: it depends what a miss costs "
                "and what a false\nalarm costs, and michi has no way to know "
                "either. Pass --cost to weigh them.",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )


def _best(rows: list[_Row], objective: str) -> _Row:
    """The row the stated objective prefers — cost is minimised, others maximised."""
    if objective == "cost":
        return min(rows, key=lambda row: row.cost)
    return max(rows, key=lambda row: getattr(row, objective))
