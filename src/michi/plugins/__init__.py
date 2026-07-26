"""Third-party extensions.

Two extension points, opened at v0.8 because their interfaces had by then
survived six milestones of real use:

- ``michi.models`` — add algorithms to the ``bench`` catalogue.
- ``michi.adapters`` — add ways to load a model for ``eval``.

Design Principles
-----------------
- Evidence-driven: an interface with one implementation is a guess, so nothing
  is opened until several concrete uses agree on its shape.
- A broken plugin is reported and skipped; michi keeps working.
- Built-ins win ties, so documented names always mean what the docs say.
- Plugin authors verify themselves against a published compatibility suite,
  which is the only way an ecosystem stays affordable for one maintainer.
"""

from __future__ import annotations

from michi.plugins.compat import check_adapter, check_model_entry
from michi.plugins.registry import (
    ADAPTER_GROUP,
    MODEL_GROUP,
    PluginError,
    PluginRecord,
    discover,
    installed_plugins,
)

__all__ = [
    "ADAPTER_GROUP",
    "MODEL_GROUP",
    "PluginError",
    "PluginRecord",
    "check_adapter",
    "check_model_entry",
    "discover",
    "installed_plugins",
]
