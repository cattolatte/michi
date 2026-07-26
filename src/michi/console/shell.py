"""The interactive console: banner, completion, and the read-eval loop.

Design Principles
-----------------
- **Completion is the reason to be here.** A one-shot CLI can never complete
  *your* column names. Once a dataset is loaded, the console can — and that
  is the capability the console exists to provide.
- **The prompt is the state display.** ``michi (train.csv → churned) ›``
  keeps the context visible at every keystroke, so nothing is ever hidden one
  command away.
- **Leaving is always safe.** Ctrl-D, ``exit``, and an interrupt all end the
  session cleanly, and unsaved context prompts a reminder rather than
  vanishing.
- The console is deletable: it dispatches to the CLI and adds no capability of
  its own.
"""

from __future__ import annotations

from collections.abc import Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    PathCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from michi import __version__
from michi.console.commands import COMMANDS, dispatch
from michi.console.session import Session

__all__ = ["MichiCompleter", "banner", "run_console"]

_SETTABLE = ("data", "target", "recipe", "runs_dir", "models", "seed", "cv")
_SHOWABLE = ("context", "columns", "models", "runs")
_FLAGGED = {"inspect", "bench", "clean", "eval", "apply", "export", "report"}
_COMMON_FLAGS = [
    "--target",
    "--explain",
    "--json",
    "--html",
    "--models",
    "--cv",
    "--seed",
    "--out",
    "--help",
]


def banner() -> str:
    """The console banner: one screen, no more."""
    return f"""
        [bold red]道[/]  [bold]michi[/] [dim]v{__version__}[/]

        [dim]a local-first ML workbench — automate implementation,
        never judgement[/]

        [dim]`help` lists commands · `use <file>` loads data ·
        Tab completes · Ctrl-D exits[/]
"""


def run_console(session: Session | None = None) -> int:
    """Run the interactive console until the user leaves.

    Parameters
    ----------
    session
        Starting context; loaded from ``michi.toml`` when omitted.

    Returns
    -------
    int
        Process exit code.
    """
    console = Console()
    state = session if session is not None else Session.from_defaults()

    if not console.is_terminal:
        console.print(
            "[red]the console needs an interactive terminal[/] — "
            "use the one-shot commands instead (michi --help)"
        )
        return 2

    console.print(banner())
    if state.data:
        console.print(f"  [dim]context restored from michi.toml — {state.data}[/]\n")

    prompt_session = _build_prompt(state)
    running = True
    while running:
        try:
            line = prompt_session.prompt(state.prompt)
        except KeyboardInterrupt:
            console.print("[dim]interrupted — Ctrl-D or `exit` to leave[/]")
            continue
        except EOFError:
            break
        if not line.strip():
            continue
        running = dispatch(line, state, console)

    if state.dirty:
        console.print(
            "\n  [dim]context not saved — `save` writes it to michi.toml next time[/]"
        )
    console.print("\n  [dim]道 — until next time[/]\n")
    return 0


def _build_prompt(session: Session) -> PromptSession[str]:
    """Build the prompt_toolkit session with history and completion."""
    return PromptSession(
        history=InMemoryHistory(),
        completer=MichiCompleter(session),
        complete_while_typing=False,
    )


class MichiCompleter(Completer):
    """Completes commands, and the user's own columns and model names.

    Completing a user's actual column names is the one thing a one-shot CLI
    can never do, and the reason the console is worth entering at all.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.paths = PathCompleter(expanduser=True)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        """Yield completions appropriate to the position in the line."""
        text = document.text_before_cursor
        words = text.split()
        leading = words[0] if words else ""

        if len(words) <= 1 and not text.endswith(" "):
            yield from self._words([item.name for item in COMMANDS], leading)
            return

        position = len(words) - (0 if text.endswith(" ") else 1)
        fragment = "" if text.endswith(" ") else words[-1]

        if leading in {"use", "apply", "export", "save"} and position == 1:
            yield from self.paths.get_completions(document, complete_event)
        elif leading == "set" and position == 1:
            yield from self._words(list(_SETTABLE), fragment)
        elif leading == "set" and position == 2 and words[1] == "target":
            yield from self._words(list(self.session.columns), fragment)
        elif leading == "set" and position == 2 and words[1] == "models":
            yield from self._words(_model_names(), fragment)
        elif leading == "unset" and position == 1:
            yield from self._words(list(_SETTABLE), fragment)
        elif leading == "show" and position == 1:
            yield from self._words(list(_SHOWABLE), fragment)
        elif leading == "help" and position == 1:
            yield from self._words([item.name for item in COMMANDS], fragment)
        elif fragment.startswith("-") and leading in _FLAGGED:
            yield from self._words(_COMMON_FLAGS, fragment)
        elif fragment and self.session.columns and leading in {"inspect", "bench"}:
            yield from self._words(list(self.session.columns), fragment)

    def _words(self, options: list[str], fragment: str) -> Iterator[Completion]:
        for option in options:
            if option.startswith(fragment):
                yield Completion(option, start_position=-len(fragment))


def _model_names() -> list[str]:
    from michi.bench import available_models

    return [entry.name for entry in available_models()]
