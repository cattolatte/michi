"""Discovering third-party extensions.

Design Principles
-----------------
- **Opened late, on evidence.** These extension points were not designed up
  front. They exist because the interfaces behind them — the model catalogue
  and the loader protocol — survived six milestones and a dozen concrete
  implementations without changing shape. An interface with one
  implementation is a guess.
- **A broken plugin is the plugin's problem.** Discovery isolates every entry
  point: one that fails to import, or returns the wrong thing, is reported and
  skipped. michi keeps working with everything else.
- **Built-ins win ties.** A plugin cannot silently replace a built-in model or
  loader, because a user reading `--list-models` must be able to trust that
  `rf` means what the documentation says.
- **Two points only.** ``michi.models`` adds algorithms; ``michi.adapters``
  adds ways to load a model. Everything else stays closed until something real
  needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "ADAPTER_GROUP",
    "MODEL_GROUP",
    "PluginError",
    "PluginRecord",
    "discover",
    "installed_plugins",
]

MODEL_GROUP = "michi.models"
"""Entry-point group for extra models in the ``bench`` catalogue."""

ADAPTER_GROUP = "michi.adapters"
"""Entry-point group for extra model loaders used by ``eval``."""


@dataclass(frozen=True, slots=True)
class PluginRecord:
    """One discovered entry point, loaded or not."""

    group: str
    name: str
    distribution: str
    loaded: bool
    error: str | None = None


class PluginError(Exception):
    """Raised when a plugin's contribution is unusable.

    Not a :class:`~michi.core.errors.MichiError`: a plugin's mistake is not
    michi's, and it must never surface as though michi failed.
    """


@lru_cache(maxsize=8)
def discover(group: str) -> tuple[tuple[str, Any, PluginRecord], ...]:
    """Load every entry point in a group, isolating failures.

    Parameters
    ----------
    group
        Entry-point group name, e.g. ``michi.models``.

    Returns
    -------
    tuple
        ``(name, object, record)`` triples for entry points that loaded, and
        records alone for those that did not — inspect
        :func:`installed_plugins` for the full picture.
    """
    from importlib.metadata import entry_points

    results: list[tuple[str, Any, PluginRecord]] = []
    try:
        points = entry_points(group=group)
    except Exception:  # third-party failure boundary
        return ()

    for point in points:
        distribution = getattr(getattr(point, "dist", None), "name", "unknown")
        try:
            loaded = point.load()
        except Exception as err:  # third-party failure boundary
            # A plugin that cannot even import must not stop michi from
            # starting; it is reported by `michi plugins` instead.
            results.append(
                (
                    point.name,
                    None,
                    PluginRecord(
                        group=group,
                        name=point.name,
                        distribution=str(distribution),
                        loaded=False,
                        error=str(err).splitlines()[0][:160],
                    ),
                )
            )
            continue
        results.append(
            (
                point.name,
                loaded,
                PluginRecord(
                    group=group,
                    name=point.name,
                    distribution=str(distribution),
                    loaded=True,
                ),
            )
        )
    return tuple(results)


def installed_plugins() -> tuple[PluginRecord, ...]:
    """Every discovered plugin across all groups, working or not."""
    records: list[PluginRecord] = []
    for group in (MODEL_GROUP, ADAPTER_GROUP):
        records.extend(record for _, _, record in discover(group))
    return tuple(records)
