"""``path`` — the stages of a project, and the command for each.

Design Principles
-----------------
- **A map, not a driver.** ``path`` prints the stages and the command that
  covers each one, and runs none of them. Its companion ``walk`` does run
  them, but only one at a time and only after asking — see
  :mod:`michi.console.walk` and ADR-0003 for the constraints that keep a
  guided sequence from becoming workflow ownership.
- **Stages are not steps.** Every entry stands alone and every entry is
  optional. A user who only wants to profile a CSV should be able to read this
  screen, run one command, and leave.
- **Status is observed, never remembered.** The marks come from looking at the
  session context and the filesystem right now. michi keeps no hidden record
  of what you have "completed", because there is no sequence to be partway
  through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["STAGES", "Stage", "render_path"]


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage of a project, and the michi command that covers it.

    Examples
    --------
    >>> STAGES[0].command
    'inspect'
    >>> STAGES[0].kanji
    '見'
    """

    kanji: str
    romaji: str
    english: str
    command: str
    usage: str


STAGES: tuple[Stage, ...] = (
    Stage("見", "miru", "see what you have", "inspect", "inspect --explain"),
    Stage("整", "totonoeru", "decide what to fix", "clean", "clean --target <col>"),
    Stage("直", "naosu", "produce the fixed data", "apply", "apply -o clean.parquet"),
    Stage("比", "kuraberu", "compare models honestly", "bench", "bench --models …"),
    Stage("確", "tashikameru", "verify one model", "eval", "eval <model.pkl>"),
    Stage("探", "sagasu", "search a grid", "sweep", "sweep sweep.yaml"),
    Stage("記", "shirusu", "write up what happened", "report", "report runs/"),
    Stage("出", "dasu", "take the code and go", "export", "export -o pipeline.py"),
)


def render_path(session: object, console: object) -> None:
    """Print the stage map, marking what the current context already has.

    Parameters
    ----------
    session
        The console :class:`~michi.console.session.Session`; its context
        decides which stages are marked ready.
    console
        A :class:`rich.console.Console` to print to.
    """
    from rich import box
    from rich.table import Table
    from rich.text import Text

    data = getattr(session, "data", None)
    target = getattr(session, "target", None)
    recipe = getattr(session, "recipe", None)
    runs_dir = getattr(session, "runs_dir", None)

    ready = _ready(data, target, recipe, runs_dir)

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=False,
        pad_edge=False,
        show_edge=False,
        padding=(0, 2, 0, 0),
    )
    table.add_column(width=1)
    # The kanji column is fixed-width and nothing hangs off its right edge:
    # 道-class characters are double-width in some terminals and single in
    # others, and a column that assumed either would be ragged on the other.
    table.add_column(width=2, style="bold red")
    table.add_column(no_wrap=True)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="dim", no_wrap=True)

    for stage in STAGES:
        runnable = stage.command in ready
        table.add_row(
            Text("✓" if runnable else "·", style="green" if runnable else "dim"),
            stage.kanji,
            Text(f"{stage.romaji} — {stage.english}"),
            stage.command,
            stage.usage,
        )

    from rich.padding import Padding

    console.print()  # type: ignore[attr-defined]
    # Padded to line up with the note underneath it, which is prose and so
    # carries the same two-space margin every other console message does.
    console.print(Padding(table, (0, 0, 0, 2)))  # type: ignore[attr-defined]
    console.print(  # type: ignore[attr-defined]
        "  [dim]✓ marks a stage this context can run right now. "
        "Every stage stands alone —[/]\n"
        "  [dim]run one and stop, or none of them. michi walks none of it "
        "for you.[/]\n"
    )


def _ready(
    data: object, target: object, recipe: object, runs_dir: object
) -> frozenset[str]:
    """Work out which stages the current context can run immediately."""
    runnable: set[str] = set()
    if data:
        runnable.update({"inspect", "clean"})
        if recipe:
            runnable.add("apply")
        if target:
            runnable.update({"bench", "eval", "sweep"})
    if recipe:
        runnable.add("export")
    if (
        runs_dir
        and Path(str(runs_dir)).is_dir()
        and any(Path(str(runs_dir)).glob("*.json"))
    ):
        runnable.add("report")
    return frozenset(runnable)
