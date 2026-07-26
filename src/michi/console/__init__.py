"""The interactive console — ``michi`` with no arguments.

An exploratory shell in the spirit of a security console: visible context,
tab completion over *your* data, and a session you can export as a script.

Design Principles
-----------------
- **Zero logic.** Every verb dispatches to the same Typer application the
  shell invokes. No capability exists only here, and deleting this package
  would remove convenience, never function.
- **Context is the ``michi.toml`` model**, held in memory and visible in the
  prompt. ``save`` writes it; nothing dies silently with the session.
- **Every session is exportable** as a replayable script of one-shot commands,
  so exploration stays reproducible.
"""

from __future__ import annotations

from michi.console.commands import COMMANDS, ConsoleCommand, dispatch, expand
from michi.console.session import Session
from michi.console.shell import banner, run_console

__all__ = [
    "COMMANDS",
    "ConsoleCommand",
    "Session",
    "banner",
    "dispatch",
    "expand",
    "run_console",
]
