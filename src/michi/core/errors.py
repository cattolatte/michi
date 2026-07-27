"""The michi error hierarchy.

Every public michi entry point raises a `MichiError` subclass so that callers
can catch a single root type. Third-party failures (pandas, sklearn, …) are
wrapped at module boundaries and chained with ``raise ... from ...`` so the
original cause is never lost.

Design Principles
-----------------
- One root exception (`MichiError`) for the whole toolkit.
- Subclasses map to domain boundaries (data, models, recipes, runs, reports),
  never to third-party libraries.
- Error messages are actionable: they state what failed and what to do next
  (e.g. the exact ``pip install`` command for a missing extra).
- Built-in ``KeyError`` / ``ValueError`` / ``IndexError`` remain appropriate
  for collection-level contracts inside modules.
"""

from __future__ import annotations

__all__ = [
    "DISTRIBUTION",
    "DataError",
    "MichiError",
    "ModelError",
    "RecipeError",
    "ReportError",
    "RunError",
    "install_hint",
]


DISTRIBUTION = "komichi"
"""The name michi is distributed under.

``michi`` was already taken on PyPI, so the distribution is ``komichi`` —
小道, a small path. The command and the import are both ``michi``.

Error messages tell users what to *install*, so they must use this name. It
lives in one constant precisely so that a rename can never leave a message
pointing at a package nobody can install.
"""


def install_hint(extra: str | None = None) -> str:
    """Return the exact command that installs michi, or one of its extras.

    Examples
    --------
    >>> install_hint("bench")
    "pip install 'komichi[bench]'"
    >>> install_hint()
    'pip install komichi'
    """
    if extra is None:
        return f"pip install {DISTRIBUTION}"
    return f"pip install '{DISTRIBUTION}[{extra}]'"


class MichiError(Exception):
    """Base class for all errors raised by michi's public API.

    Examples
    --------
    >>> try:
    ...     raise DataError("could not parse data.csv")
    ... except MichiError as err:
    ...     print(err)
    could not parse data.csv
    """


class DataError(MichiError):
    """Raised when a dataset cannot be read, parsed, or validated."""


class ModelError(MichiError):
    """Raised when a model cannot be loaded or lacks the predict protocol."""


class RecipeError(MichiError):
    """Raised when a recipe is invalid or incompatible with the given data."""


class RunError(MichiError):
    """Raised when an evaluation, benchmark, or sweep run fails."""


class ReportError(MichiError):
    """Raised when a report cannot be rendered from its inputs."""
