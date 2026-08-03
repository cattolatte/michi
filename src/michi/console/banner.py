"""The console banner: block art, a live inventory, and a rotating tip.

Design Principles
-----------------
- **The inventory is counted, never typed.** The verb, model, op, and
  explanation counts are read from the registries at startup, so a banner that
  claims fourteen models is a banner backed by fourteen models. A hand-written
  number would be wrong the first time someone adds a plugin.
- **Startup stays instant.** Every registry consulted here is a plain table of
  names and factories; none of them import pandas, sklearn, or a plotting
  library. The banner costs milliseconds, and a console that takes a second to
  appear is a console people stop opening.
- **Tips are content, not code.** They live in ``explain/content/tips.yaml``
  and are edited like documentation.
- **Colour is Rich's problem.** Rich already suppresses styling for a pipe,
  a dumb terminal, and ``NO_COLOR``; re-implementing that with raw escapes
  would be a second, worse copy of it.
"""

from __future__ import annotations

import random
from functools import lru_cache

from michi import __version__

__all__ = ["banner", "inventory", "tip", "tips"]

# ANSI Shadow block capitals. Bundled as text rather than generated: a figlet
# dependency to render five fixed letters would be a dependency to render five
# fixed letters.
_ART = r"""
███╗   ███╗██╗ ██████╗██╗  ██╗██╗
████╗ ████║██║██╔════╝██║  ██║██║
██╔████╔██║██║██║     ███████║██║
██║╚██╔╝██║██║██║     ██╔══██║██║
██║ ╚═╝ ██║██║╚██████╗██║  ██║██║
╚═╝     ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝
""".strip("\n")

# The seal sits at the end of its line, never inside a box or a column. 道 is
# double-width in some terminals and single in others; anything aligned after
# it would look correct here and ragged on someone else's machine.
_SEAL = "[bold white on red] 道 [/]"

_WIDTH = 66
"""Inner width of the inventory brackets, in characters."""


def inventory() -> tuple[str, ...]:
    """Count what this installation actually offers.

    Returns
    -------
    tuple of str
        Lines for the bracketed block, longest-lived facts first.

    Examples
    --------
    >>> lines = inventory()
    >>> "verbs" in lines[1]
    True
    """
    from michi.bench import available_models
    from michi.cli.app import app
    from michi.console.commands import COMMANDS
    from michi.explain.registry import explanations
    from michi.recipes.model import known_operations

    verbs = sum(1 for command in app.registered_commands if command.name)
    return (
        f"michi v{__version__}  ·  a local-first ML workbench",
        f"{verbs} verbs  ·  {len(available_models())} models  ·  "
        f"{len(known_operations())} recipe ops  ·  {len(explanations())} explanations",
        f"{len(COMMANDS)} console commands  ·  "
        "no account, no telemetry, no network call",
    )


@lru_cache(maxsize=1)
def tips() -> tuple[str, ...]:
    """Load the bundled console tips.

    Examples
    --------
    >>> len(tips()) > 3
    True
    """
    from importlib.resources import files

    import yaml

    source = files("michi.explain.content").joinpath("tips.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or []
    return tuple(" ".join(str(item).split()) for item in payload)


def tip(rng: random.Random | None = None) -> str:
    """Pick one tip to show under the banner.

    Parameters
    ----------
    rng
        Source of randomness; pass a seeded instance for a reproducible pick.

    Examples
    --------
    >>> import random
    >>> tip(random.Random(0)) in tips()
    True
    """
    chooser = rng if rng is not None else random
    return chooser.choice(tips())


def banner(rng: random.Random | None = None) -> str:
    """The console banner, as Rich markup.

    Parameters
    ----------
    rng
        Passed through to :func:`tip` so the whole banner can be made
        reproducible in a test.

    Examples
    --------
    >>> import random
    >>> "michi" in banner(random.Random(0))
    True
    """
    art = _ART.splitlines()
    # The seal rides the third row, where the block letters leave the most
    # white space to its left.
    art[2] = f"{art[2]}     {_SEAL}"
    rendered = "\n".join(f"  [bold red]{line}[/]" for line in art)
    # Rows carrying the seal must not be swallowed by the red span around them.
    rendered = rendered.replace(f"     {_SEAL}[/]", f"[/]     {_SEAL}")

    lines = inventory()
    head = f"        [dim]=[[/] {lines[0].ljust(_WIDTH)} [dim]][/]"
    rest = "\n".join(
        f"[dim]+ -- --=[[/] {line.ljust(_WIDTH)} [dim]][/]" for line in lines[1:]
    )

    return f"""
{rendered}

{head}
{rest}

  [dim]心得[/]  [dim]michi lists the options; you choose.[/]
  [dim]path[/] [dim]walks the stages[/] · [dim]help[/] [dim]lists commands[/] · \
[dim]use <file>[/] [dim]loads data[/]

{_tip_block(tip(rng))}
"""


def _tip_block(text: str) -> str:
    """Wrap a tip under a hanging indent, so it reads as one sentence."""
    import textwrap

    lines = textwrap.wrap(text, width=_WIDTH + 6)
    body = "\n".join(
        f"  [dim]{'     ' if index else 'tip: '}{line}[/]"
        for index, line in enumerate(lines)
    )
    return body
