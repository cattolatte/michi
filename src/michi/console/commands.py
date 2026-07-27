"""Console commands — a skin over the one-shot CLI.

Design Principles
-----------------
- **Zero logic.** Every verb builds an argument list and hands it to the same
  Typer application the shell would invoke. There is no second code path, so
  the console can never behave differently from the command line, and no
  capability can exist only here.
- **Context fills the flags you did not type.** ``bench`` in the console
  expands to the full command, which is exactly what ``history --export``
  writes out.
- **Everything is inspectable.** ``show context`` prints the state,
  ``history`` prints what you ran, and both are one word away at all times.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from michi.console.session import Session
from michi.core.errors import install_hint

__all__ = ["COMMANDS", "ConsoleCommand", "dispatch", "expand", "split_line"]


@dataclass(frozen=True, slots=True)
class ConsoleCommand:
    """One console command: what it does and how it is described."""

    name: str
    summary: str
    usage: str
    group: str


COMMANDS: tuple[ConsoleCommand, ...] = (
    ConsoleCommand(
        "help", "show this help, or help on one command", "help [command]", "session"
    ),
    ConsoleCommand(
        "path",
        "list the stages of a project and the command for each",
        "path",
        "path",
    ),
    ConsoleCommand(
        "walk",
        "go stage by stage, asking before each — nothing runs unprompted",
        "walk [stage]",
        "path",
    ),
    ConsoleCommand(
        "use", "load a dataset into the session context", "use <path>", "context"
    ),
    ConsoleCommand("set", "set a context value", "set <key> <value>", "context"),
    ConsoleCommand("unset", "clear a context value", "unset <key>", "context"),
    ConsoleCommand(
        "show",
        "show context, columns, models, or runs",
        "show [context|columns|models|runs]",
        "context",
    ),
    ConsoleCommand("inspect", "profile the loaded dataset", "inspect [flags]", "verbs"),
    ConsoleCommand(
        "eval",
        "evaluate a model on the loaded dataset",
        "eval <model> [flags]",
        "verbs",
    ),
    ConsoleCommand(
        "bench", "compare models on the loaded dataset", "bench [flags]", "verbs"
    ),
    ConsoleCommand("clean", "author a cleaning recipe", "clean [flags]", "verbs"),
    ConsoleCommand(
        "apply", "apply a recipe to the loaded dataset", "apply [flags]", "verbs"
    ),
    ConsoleCommand(
        "export", "compile a recipe into pipeline code", "export [flags]", "verbs"
    ),
    ConsoleCommand(
        "sweep", "run a recorded experiment grid", "sweep <plan.yaml>", "verbs"
    ),
    ConsoleCommand("report", "render recorded runs", "report [flags]", "verbs"),
    ConsoleCommand(
        "history",
        "show this session's commands, or export them as a script",
        "history [--export <file>]",
        "session",
    ),
    ConsoleCommand("save", "write the context to michi.toml", "save [path]", "session"),
    ConsoleCommand("clear", "clear the screen", "clear", "session"),
    ConsoleCommand("exit", "leave the console (also: quit, Ctrl-D)", "exit", "session"),
)

_VERBS = {
    "inspect",
    "eval",
    "bench",
    "clean",
    "apply",
    "export",
    "report",
    "sweep",
}
_SETTABLE = ("data", "target", "recipe", "runs_dir", "models", "seed", "cv")


def split_line(line: str) -> list[str]:
    """Split a console line into arguments, treating backslashes literally.

    ``shlex.split`` reads a backslash as an escape character, which silently
    mangles every Windows path a user types (``use C:\\data\\train.csv``
    becomes ``C:datatrain.csv``). Quoting still works, so a path with spaces
    can be given as ``use "my file.csv"``.

    Examples
    --------
    >>> split_line(r"use C:\\data\train.csv")
    ['use', 'C:\\data\\train.csv']
    >>> split_line('use "my file.csv"')
    ['use', 'my file.csv']
    """
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    return list(lexer)


def dispatch(line: str, session: Session, console: Console) -> bool:
    """Run one console line. Returns ``False`` when the session should end."""
    try:
        parts = split_line(line)
    except ValueError as err:
        console.print(f"[red]could not parse that line:[/] {err}")
        return True
    if not parts:
        return True

    name, args = parts[0], parts[1:]
    handler: Callable[[list[str], Session, Console], bool] | None = _HANDLERS.get(name)
    if handler is not None:
        session.record(line)
        return handler(args, session, console)

    if name in _VERBS:
        session.record(line)
        _run_verb(name, args, session, console)
        return True

    console.print(
        f"[red]unknown command[/] {name!r} — type [bold]help[/] to see what exists"
    )
    return True


def expand(name: str, args: list[str], session: Session) -> list[str]:
    """Build the full one-shot argument list for a console verb.

    This is what makes the console a skin: the expansion below is exactly the
    command a user would type in a shell, and exactly what ``history
    --export`` writes into the script.
    """
    argv: list[str] = [name]
    flags = set(args)

    def missing(*names: str) -> bool:
        return not any(flag in flags for flag in names)

    if name in {"inspect", "bench", "clean"}:
        if session.data and not _has_positional(args):
            argv.append(session.data)
        if session.target and missing("--target", "-t"):
            argv += ["--target", session.target]
    elif name == "eval":
        if session.data and len(_positionals(args)) < 2:
            argv += args
            argv.append(session.data)
            args = []
        if session.target and missing("--target", "-t"):
            argv += ["--target", session.target]
    elif name == "apply":
        if session.recipe and not _has_positional(args):
            argv.append(session.recipe)
        if session.data and len(_positionals(args)) < 2:
            argv.append(session.data)
    elif name == "export":
        if session.recipe and not _has_positional(args):
            argv.append(session.recipe)
    elif name == "report" and not _has_positional(args):
        argv.append(session.runs_dir)

    if name == "bench":
        if missing("--models", "-m"):
            argv += ["--models", session.models]
        if missing("--cv"):
            argv += ["--cv", str(session.cv)]
    if name in {"eval", "bench"} and missing("--runs-dir"):
        argv += ["--runs-dir", session.runs_dir]
    if name in {"inspect", "eval", "bench", "clean"} and missing("--seed"):
        argv += ["--seed", str(session.seed)]

    return argv + args


def _run_verb(name: str, args: list[str], session: Session, console: Console) -> None:
    """Invoke a verb through the real CLI application.

    This is the whole of the console's dispatch: the same Typer application
    the shell invokes, called with the arguments the context filled in.
    """
    import typer

    from michi.cli.app import app

    argv = expand(name, args, session)
    try:
        app(args=argv, standalone_mode=False)
    except (typer.Exit, typer.Abort, SystemExit):
        # A verb finishing — including a non-zero gate result — is not a
        # reason to end the session.
        pass
    except Exception as err:  # third-party failure boundary
        # Usage errors carry a formatted message; anything else is reported
        # as-is. Either way the console survives so the user can retry.
        formatter = getattr(err, "format_message", None)
        message = formatter() if callable(formatter) else str(err)
        console.print(f"[red]error[/] {message}")


def _handle_help(args: list[str], session: Session, console: Console) -> bool:
    if args:
        wanted = args[0]
        entry = next((item for item in COMMANDS if item.name == wanted), None)
        if entry is None:
            console.print(f"[red]no such command[/] {wanted!r}")
            return True
        console.print()
        console.print(f"  [bold]{entry.name}[/]  {entry.summary}")
        console.print(f"  [dim]usage:[/] {entry.usage}")
        if entry.name in _VERBS:
            console.print(
                f"  [dim]flags:[/] every flag of `michi {entry.name}` works here; "
                f"run `michi {entry.name} --help` in a shell for the full list"
            )
        console.print()
        return True

    console.print()
    for group, title in (
        ("path", "道 — the path, if you want one"),
        ("context", "Context — what the verbs inherit"),
        ("verbs", "Verbs — the same commands as the shell"),
        ("session", "Session"),
    ):
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=False,
            pad_edge=False,
            show_edge=False,
            title=f"  {title}",
            title_style="bold",
            title_justify="left",
        )
        table.add_column("usage", no_wrap=True, style="cyan")
        table.add_column("summary", overflow="fold")
        for entry in COMMANDS:
            if entry.group == group:
                table.add_row(f"  {entry.usage}", entry.summary)
        console.print(table)
    console.print(
        "  [dim]Tab completes commands, columns, and model names. "
        "Ctrl-D or `exit` leaves.[/]\n"
    )
    return True


def _handle_use(args: list[str], session: Session, console: Console) -> bool:
    if not args:
        console.print("[red]usage:[/] use <path>")
        return True
    session.data = args[0]
    session.dirty = True
    message = session.load_columns()
    if message and message.startswith(("no such", "could not")):
        console.print(f"[red]{message}[/]")
    elif message:
        console.print(f"[dim]{message}[/]")
    return True


def _handle_set(args: list[str], session: Session, console: Console) -> bool:
    if len(args) < 2:
        console.print(f"[red]usage:[/] set <{'|'.join(_SETTABLE)}> <value>")
        return True
    key, value = args[0], " ".join(args[1:])
    if key not in _SETTABLE:
        console.print(f"[red]unknown setting[/] {key!r} — try: {', '.join(_SETTABLE)}")
        return True

    if key in {"seed", "cv"}:
        try:
            setattr(session, key, int(value))
        except ValueError:
            console.print(f"[red]{key} must be a whole number[/] (got {value!r})")
            return True
    elif key == "data":
        return _handle_use([value], session, console)
    else:
        if key == "target" and session.columns and value not in session.columns:
            console.print(
                f"[yellow]note:[/] {value!r} is not a column of the loaded data"
            )
        setattr(session, key, value)
    session.dirty = True
    console.print(f"[dim]{key} = {value}[/]")
    return True


def _handle_unset(args: list[str], session: Session, console: Console) -> bool:
    if not args or args[0] not in _SETTABLE:
        console.print(f"[red]usage:[/] unset <{'|'.join(_SETTABLE)}>")
        return True
    key = args[0]
    defaults = Session()
    setattr(session, key, getattr(defaults, key))
    if key == "data":
        session.columns = ()
    session.dirty = True
    console.print(f"[dim]{key} cleared[/]")
    return True


def _handle_show(args: list[str], session: Session, console: Console) -> bool:
    what = args[0] if args else "context"
    if what in {"context", "options"}:
        table = Table(
            box=box.SIMPLE_HEAD,
            header_style="bold dim",
            pad_edge=False,
            show_edge=False,
        )
        table.add_column("setting", no_wrap=True)
        table.add_column("value", overflow="fold")
        for key, value in session.settings().items():
            table.add_row(
                Text(key, style="cyan"),
                Text(
                    "—" if value is None else str(value),
                    style="dim" if value is None else "",
                ),
            )
        console.print()
        console.print(table)
        if session.dirty:
            console.print("  [dim]unsaved — `save` writes these to michi.toml[/]")
        console.print()
    elif what == "columns":
        if not session.columns:
            console.print("[dim]no dataset loaded — try `use data.csv`[/]")
            return True
        console.print()
        for index, name in enumerate(session.columns, start=1):
            marker = " [green]← target[/]" if name == session.target else ""
            console.print(f"  [dim]{index:>3}[/]  {name}{marker}")
        console.print()
    elif what == "models":
        from michi.bench import available_models

        console.print()
        for entry in available_models():
            note = f" (needs {install_hint(entry.extra)})" if entry.extra else ""
            console.print(f"  [cyan]{entry.name:<12}[/] {escape(entry.summary + note)}")
        console.print()
    elif what == "runs":
        directory = Path(session.runs_dir)
        manifests = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if not manifests:
            console.print(f"[dim]no runs recorded in {directory}[/]")
            return True
        console.print()
        for path in manifests[-20:]:
            console.print(f"  [dim]{path.name}[/]")
        console.print(f"\n  [dim]{len(manifests)} total — `report` renders them[/]\n")
    else:
        console.print(
            f"[red]cannot show[/] {what!r} — try: context, columns, models, runs"
        )
    return True


def _handle_history(args: list[str], session: Session, console: Console) -> bool:
    if "--export" in args:
        index = args.index("--export")
        if index + 1 >= len(args):
            console.print("[red]usage:[/] history --export <file>")
            return True
        destination = Path(args[index + 1])
        destination.write_text(_as_script(session), encoding="utf-8")
        console.print(
            f"  [dim]wrote[/] {destination} — a replayable script of one-shot "
            "michi commands\n"
        )
        return True

    if not session.history:
        console.print("[dim]nothing run yet[/]")
        return True
    console.print()
    for index, item in enumerate(session.history, start=1):
        console.print(f"  [dim]{index:>3}[/]  {item}")
    console.print("\n  [dim]`history --export session.sh` writes it as a script[/]\n")
    return True


def _as_script(session: Session) -> str:
    """Render the session as a shell script of one-shot michi commands."""
    lines = [
        "#!/usr/bin/env bash",
        "# Recorded from a michi console session.",
        "# Every line is a plain one-shot command — the console adds nothing.",
        "set -euo pipefail",
        "",
    ]
    for item in session.history:
        parts = split_line(item)
        if not parts or parts[0] not in _VERBS:
            continue
        argv = expand(parts[0], parts[1:], session)
        lines.append("michi " + " ".join(shlex.quote(part) for part in argv))
    if len(lines) == 5:
        lines.append("# (no verbs were run in this session)")
    return "\n".join(lines) + "\n"


def _handle_save(args: list[str], session: Session, console: Console) -> bool:
    destination = Path(args[0]) if args else None
    path = session.save(destination)
    console.print(f"  [dim]wrote[/] {path}\n")
    return True


def _handle_clear(args: list[str], session: Session, console: Console) -> bool:
    console.clear()
    return True


def _handle_exit(args: list[str], session: Session, console: Console) -> bool:
    return False


def _handle_path(args: list[str], session: Session, console: Console) -> bool:
    from michi.console.path import render_path

    render_path(session, console)
    return True


def _handle_walk(args: list[str], session: Session, console: Console) -> bool:
    from michi.console.walk import run_walk

    run_walk(session, console, start=args[0] if args else None)
    return True


_HANDLERS: dict[str, Callable[[list[str], Session, Console], bool]] = {
    "help": _handle_help,
    "?": _handle_help,
    "path": _handle_path,
    "walk": _handle_walk,
    "use": _handle_use,
    "set": _handle_set,
    "unset": _handle_unset,
    "show": _handle_show,
    "history": _handle_history,
    "save": _handle_save,
    "clear": _handle_clear,
    "exit": _handle_exit,
    "quit": _handle_exit,
}


def _positionals(args: list[str]) -> list[str]:
    """Arguments that are not flags or flag values."""
    result: list[str] = []
    skip = False
    for index, item in enumerate(args):
        if skip:
            skip = False
            continue
        if item.startswith("-"):
            following = args[index + 1] if index + 1 < len(args) else ""
            skip = bool(following) and not following.startswith("-")
            continue
        result.append(item)
    return result


def _has_positional(args: list[str]) -> bool:
    return bool(_positionals(args))
