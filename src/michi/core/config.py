"""Project defaults, as a file you can read.

Design Principles
-----------------
- **Nothing is remembered invisibly.** The pain of retyping ``--target y`` on
  every command is real, but a hidden session that remembers it is worse: it
  breaks reproducibility, scripting, and CI, and produces bug reports that
  begin "it worked yesterday". michi's answer is an optional, visible
  ``michi.toml`` you can read, edit, diff, and check into git.
- **Defaults for flags only.** ``michi.toml`` supplies default *values*; it
  never changes what a command does.
- **Explicit always wins.** Precedence is flags > ``michi.toml`` > built-in
  defaults, and ``michi info`` prints where each value came from, so a
  surprising value is never a mystery.
- Absent, empty, or malformed config degrades to built-in defaults with a
  warning rather than failing a command.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from michi.core.errors import MichiError

__all__ = ["CONFIG_FILENAME", "ProjectDefaults", "find_config", "load_defaults"]

CONFIG_FILENAME = "michi.toml"
"""Name of the optional project defaults file."""

_KEYS = ("data", "target", "recipe", "runs_dir", "models", "seed", "cv")


@dataclass(frozen=True, slots=True)
class ProjectDefaults:
    """Default flag values read from ``michi.toml``.

    Examples
    --------
    >>> defaults = ProjectDefaults(target="churned")
    >>> defaults.resolve("target", None)
    'churned'
    >>> defaults.resolve("target", "override")
    'override'
    """

    data: str | None = None
    target: str | None = None
    recipe: str | None = None
    runs_dir: str | None = None
    models: str | None = None
    seed: int | None = None
    cv: int | None = None
    source: Path | None = field(default=None, compare=False)

    @property
    def is_empty(self) -> bool:
        """Whether the file supplied no values at all."""
        return all(getattr(self, key) is None for key in _KEYS)

    def resolve(self, key: str, explicit: Any) -> Any:
        """Return the explicit value if given, else the configured default."""
        if explicit is not None:
            return explicit
        return getattr(self, key, None)

    def origin(self, key: str, explicit: Any) -> str:
        """Where a value came from, for ``michi info``."""
        if explicit is not None:
            return "flag"
        return CONFIG_FILENAME if getattr(self, key, None) is not None else "built-in"

    def with_values(self, **values: Any) -> ProjectDefaults:
        """Return a copy with some values changed."""
        return replace(self, **values)

    def to_toml(self) -> str:
        """Render as a commented ``michi.toml``."""
        lines = [
            "# michi project defaults.",
            "#",
            "# These are default values for command-line flags, nothing more.",
            "# An explicit flag always wins, and `michi info` shows where each",
            "# value came from. Delete this file and michi behaves normally.",
            "",
            "[defaults]",
        ]
        for key in _KEYS:
            value = getattr(self, key)
            if value is None:
                continue
            rendered = str(value) if isinstance(value, int) else f'"{value}"'
            lines.append(f"{key} = {rendered}")
        return "\n".join(lines) + "\n"

    def write(self, destination: Path) -> None:
        """Write the defaults file."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_toml(), encoding="utf-8")


def find_config(start: Path | None = None) -> Path | None:
    """Find ``michi.toml`` in a directory or any parent.

    Searching upward means a command run from a subdirectory still finds the
    project's defaults, which is what users expect from tool configuration.
    """
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        config = candidate / CONFIG_FILENAME
        if config.is_file():
            return config
    return None


def load_defaults(start: Path | None = None) -> ProjectDefaults:
    """Load project defaults, or return empty defaults if there are none.

    Raises
    ------
    MichiError
        If the file exists but cannot be parsed — a broken config is worth
        reporting, because silently ignoring it would be more confusing than
        failing.
    """
    import tomllib

    config = find_config(start)
    if config is None:
        return ProjectDefaults()

    try:
        payload: dict[str, Any] = tomllib.loads(config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as err:
        msg = f"could not parse {config}: {err}"
        raise MichiError(msg) from err
    except OSError as err:
        msg = f"could not read {config}: {err}"
        raise MichiError(msg) from err

    section = payload.get("defaults", payload)
    if not isinstance(section, dict):
        return ProjectDefaults(source=config)

    def _text(key: str) -> str | None:
        value = section.get(key)
        return None if value is None else str(value)

    def _number(key: str) -> int | None:
        value = section.get(key)
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    return ProjectDefaults(
        data=_text("data"),
        target=_text("target"),
        recipe=_text("recipe"),
        runs_dir=_text("runs_dir"),
        models=_text("models"),
        seed=_number("seed"),
        cv=_number("cv"),
        source=config,
    )
