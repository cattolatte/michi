"""Reporting errors from the command line.

Design Principles
-----------------
- **An error message is data, not markup.** michi's own messages name package
  extras like ``michi-ml[bench]``, and third-party messages can contain
  anything at all. Rendering either as rich markup silently swallows the
  square brackets — which turns an actionable install command into a broken
  one. Messages are therefore escaped, and only michi's own prefix is styled.
- One helper, so every command reports failures identically.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

__all__ = ["fail", "warn"]


def fail(message: str) -> None:
    """Print an error to stderr, with the message rendered verbatim."""
    Console(stderr=True).print(f"[bold red]error[/] {escape(message)}")


def warn(console: Console, message: str) -> None:
    """Print a note to the console, with the message rendered verbatim."""
    console.print(f"  [dim]{escape(message)}[/]")
