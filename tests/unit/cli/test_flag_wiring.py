"""Every declared flag must actually reach the code behind it.

Four flags have shipped declared-but-unwired in this project: `eval
--importance`, `bench --oof`, `tune --metric`, and `tune --group`. Each
appeared in `--help`, accepted a value, and silently did nothing — the kind of
defect a normal test suite cannot see, because there is nothing to assert
against a feature that was never connected.

This walks every command's signature and checks each parameter is referenced
somewhere in its own body. It is crude, and it has caught four real bugs.
"""

from __future__ import annotations

import inspect
import re

from michi.cli.app import app

# Parameters consumed by name in the signature and never mentioned again: a
# positional argument passed straight through, or one read via `defaults`.
_PASSTHROUGH = {
    "data",
    "model",
    "model_path",
    "baseline",
    "current",
    "recipe_path",
}


def test_no_command_declares_a_flag_it_never_uses() -> None:
    """A flag in --help that does nothing is worse than a missing flag.

    A missing flag fails loudly the moment someone types it. A dead one
    accepts the value, prints no warning, and quietly produces the result the
    user was trying to change.
    """
    dead: list[str] = []
    for command in app.registered_commands:
        if not command.name:
            continue
        source = inspect.getsource(command.callback)
        body = source.split(") -> None:", 1)[-1]
        for name in inspect.signature(command.callback).parameters:
            if name in _PASSTHROUGH:
                continue
            if not re.search(rf"\b{re.escape(name)}\b", body):
                dead.append(f"{command.name} --{name.rstrip('_').replace('_', '-')}")
    assert dead == [], f"declared but never used: {dead}"


def test_every_verb_is_documented() -> None:
    """A verb documented only in release notes is a verb nobody finds.

    Five shipped that way before 2.0 froze them, which is the wrong order:
    the freeze is a promise about things people can discover.
    """
    from pathlib import Path

    docs = " ".join(
        path.read_text(encoding="utf-8")
        for path in Path(__file__).parents[3].joinpath("docs").glob("*.md")
    )
    missing = [
        command.name
        for command in app.registered_commands
        if command.name and f"michi {command.name}" not in docs
    ]
    assert missing == [], f"undocumented verbs: {missing}"


def test_every_recipe_operation_is_reachable_from_a_flag() -> None:
    """Flag parity is absolute: nothing may exist only interactively."""
    from pathlib import Path

    from michi.recipes.model import _KNOWN_OPS

    author = (
        Path(__file__)
        .parents[3]
        .joinpath("src/michi/recipes/author.py")
        .read_text(encoding="utf-8")
    )
    missing = [op for op in sorted(_KNOWN_OPS) if f'"{op}"' not in author]
    assert missing == [], f"operations with no flag path: {missing}"
