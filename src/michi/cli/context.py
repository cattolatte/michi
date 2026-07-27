"""Resolving flag values against project defaults.

Design Principles
-----------------
- **One resolution path for every command.** ``michi.toml`` is documented as
  supplying defaults for flags; that promise is only true if every command
  honours it identically, so they all resolve through here.
- **Explicit always wins.** A flag the user typed is never overridden, and a
  value that came from a file is reported when it matters, so a surprising
  result is always traceable to something readable.
- A missing or malformed config degrades to built-in defaults rather than
  failing a command that would otherwise have worked.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from michi.core.config import ProjectDefaults, load_defaults
from michi.core.errors import DataError, MichiError

__all__ = ["Defaults", "resolve_defaults"]

T = TypeVar("T")


class Defaults:
    """Flag defaults for one command invocation.

    Wraps :class:`~michi.core.config.ProjectDefaults` with the resolution
    helpers commands need, so no command has to know where a value came from.
    """

    __slots__ = ("_defaults",)

    def __init__(self, defaults: ProjectDefaults) -> None:
        self._defaults = defaults

    @property
    def source(self) -> Path | None:
        """The config file these defaults came from, if any."""
        return self._defaults.source

    def text(self, key: str, explicit: str | None) -> str | None:
        """Resolve a string flag."""
        value = self._defaults.resolve(key, explicit)
        return None if value is None else str(value)

    def number(self, key: str, explicit: int | None) -> int | None:
        """Resolve an integer flag."""
        value = self._defaults.resolve(key, explicit)
        return None if value is None else int(value)

    def path(self, key: str, explicit: Path | None) -> Path | None:
        """Resolve a path flag."""
        value = self._defaults.resolve(key, explicit)
        return None if value is None else Path(str(value))

    def target_for(
        self, explicit: str | None, columns: Iterable[str]
    ) -> tuple[str | None, str | None]:
        """Resolve the target column against the data actually loaded.

        A typed ``--target`` is a command: if the column is absent, that is an
        error the user needs to see. A configured target is only a hint, and a
        hint that cannot apply — because this run is over a different dataset —
        must not fail a command that would otherwise have worked. Dropping it
        is reported rather than silent.

        Returns
        -------
        tuple
            The target to use, and a note to print when a configured target
            was set aside.
        """
        if explicit is not None:
            return explicit, None

        configured = self._defaults.target
        if configured is None or configured in set(columns):
            return configured, None
        return None, (
            f"michi.toml sets target {configured!r}, which this data does not "
            "have — continuing without a target"
        )

    def required_data(self, explicit: Path | None) -> Path:
        """Resolve the dataset, which every data verb needs from somewhere.

        Raises
        ------
        DataError
            If no dataset was given and none is configured, with a message
            naming both ways to supply one.
        """
        resolved = self.path("data", explicit)
        if resolved is None:
            msg = (
                "no dataset given. Pass one as an argument, or set "
                '`data = "path/to/file.csv"` under [defaults] in michi.toml.'
            )
            raise DataError(msg)
        return resolved


def resolve_defaults(start: Path | None = None) -> Defaults:
    """Load project defaults, tolerating a broken or absent config.

    A malformed ``michi.toml`` is reported by ``michi info``; it must not stop
    a command that was given everything it needs on the command line.
    """
    try:
        return Defaults(load_defaults(start))
    except MichiError:
        return Defaults(ProjectDefaults())
