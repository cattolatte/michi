"""``walk`` — the stages, one at a time, asking before each.

Design Principles
-----------------
- **It asks; it never suggests.** Every stage offers run / skip / stop, and
  the prompts collect facts michi cannot know (which column is the target,
  which models to compare). No stage tells the user which choice is better,
  because that is the judgement michi exists not to make.
- **Order is mechanical, not editorial.** ``apply`` follows ``clean`` because
  a recipe has to exist before it can be applied, not because michi thinks
  cleaning matters more than benchmarking. Any stage may be skipped, and the
  walk may be entered or left at any point.
- **Nothing happens that you could not have typed.** Each stage prints the
  one-shot command before running it, dispatches through the same CLI as the
  shell, and records it in history — so a walk exports to a script exactly
  like a hand-typed session.
- **A walk leaves no residue.** It ends with the same context, artifacts, and
  history a user would have from running the commands themselves. There is no
  "walk state" to resume, because there is no sequence to be partway through.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from michi.console.path import STAGES, Stage
from michi.console.session import Session

__all__ = ["run_walk"]

_STOP = "stop"
_SKIP = "skip"
_RUN = "run"


@dataclass(frozen=True, slots=True)
class _Need:
    """A fact a stage cannot run without, and how to ask for it."""

    key: str
    prompt: str


# What each stage needs before it can run at all. Anything already in the
# context is never asked for twice.
_NEEDS: dict[str, tuple[_Need, ...]] = {
    "inspect": (_Need("data", "Which dataset?"),),
    "clean": (_Need("data", "Which dataset?"),),
    "apply": (_Need("recipe", "Which recipe?"),),
    "bench": (
        _Need("data", "Which dataset?"),
        _Need("target", "Which column is the target?"),
    ),
    "eval": (_Need("data", "Which dataset?"),),
    "sweep": (_Need("data", "Which dataset?"),),
    "report": (),
    "export": (_Need("recipe", "Which recipe?"),),
}


def run_walk(session: Session, console: Console, *, start: str | None = None) -> None:
    """Walk the stages from `start`, asking what to do at each.

    Parameters
    ----------
    session
        The console context; stages read from it and write back to it.
    console
        Where to print.
    start
        Stage command to begin at. Defaults to the first stage.
    """
    stages = list(STAGES)
    if start:
        names = [stage.command for stage in stages]
        if start not in names:
            console.print(
                f"[red]no such stage[/] {start!r} — stages are {', '.join(names)}"
            )
            return
        stages = stages[names.index(start) :]

    _header(console, len(stages))

    for number, stage in enumerate(stages, start=1):
        choice = _offer(stage, number, len(stages), session, console)
        if choice == _STOP:
            console.print("\n  [dim]stopped — everything up to here is on disk.[/]\n")
            return
        if choice == _SKIP:
            continue
        _perform(stage, session, console)

    console.print(
        "\n  [dim]道 — end of the path. `history --export` writes it out.[/]\n"
    )


def _header(console: Console, total: int) -> None:
    """Say what a walk is before the first question."""
    text = Text()
    text.append(" 道 ", style="bold white on red")
    text.append("  michi walk", style="bold")
    text.append(f"  ·  {total} stages", style="dim")
    console.print()
    console.print(text)
    console.print(
        "\n  [dim]Every stage asks before it runs, and every stage is optional.[/]\n"
        "  [dim]michi never picks for you — it only does what you say, and prints[/]\n"
        "  [dim]the command it ran so you can run it yourself next time.[/]"
    )


def _offer(
    stage: Stage, number: int, total: int, session: Session, console: Console
) -> str:
    """Show one stage and ask what to do with it."""
    import questionary

    heading = Text()
    heading.append(f"\n  [{number}/{total}]  ", style="dim")
    heading.append(f"{stage.kanji} ", style="bold red")
    heading.append(f"{stage.romaji}", style="bold")
    heading.append(f" — {stage.english}", style="none")
    console.print(heading)
    console.print(f"      [dim]covered by[/] [cyan]michi {stage.usage}[/]")

    missing = [need for need in _NEEDS[stage.command] if not _value(session, need.key)]
    note = (
        f"      [dim]needs {', '.join(need.key for need in missing)}[/]"
        if missing
        else ""
    )
    if note:
        console.print(note)

    answer = questionary.select(
        "  What would you like to do?",
        choices=[
            questionary.Choice(title=f"run {stage.command}", value=_RUN),
            questionary.Choice(title="skip this stage", value=_SKIP),
            questionary.Choice(title="stop walking", value=_STOP),
        ],
        default=_SKIP,
        instruction=" ",
    ).ask()
    # Ctrl-C at the prompt returns None; treat it as leaving, not as skipping.
    return str(answer) if answer is not None else _STOP


def _perform(stage: Stage, session: Session, console: Console) -> None:
    """Collect what the stage needs, then run it through the real CLI."""
    import questionary

    # Setting context goes through the same handler `set` uses, so a walk
    # cannot put the session into a state a typed command could not.
    from michi.console.commands import _handle_set, _run_verb, expand

    for need in _NEEDS[stage.command]:
        if _value(session, need.key):
            continue
        given = questionary.text(f"  {need.prompt}").ask()
        if not given or not given.strip():
            console.print("      [dim]nothing given — skipping this stage.[/]")
            return
        _handle_set([need.key, given.strip()], session, console)

    extra = questionary.text("  Extra flags? (blank for none)").ask()
    args = _split(extra)

    argv = expand(stage.command, args, session)
    console.print(f"\n  [dim]$[/] [cyan]michi {' '.join(argv)}[/]\n")
    session.record(f"{stage.command} {' '.join(args)}".strip())
    _run_verb(stage.command, args, session, console)


def _split(value: str | None) -> list[str]:
    """Split typed flags the same way the console splits a typed line."""
    from michi.console.commands import split_line

    if not value or not value.strip():
        return []
    try:
        return split_line(value)
    except ValueError:
        return value.split()


def _value(session: Session, key: str) -> object:
    """Read one context value by name."""
    return getattr(session, key, None)
