"""Core: shared foundations for every michi verb.

This package will hold the artifact value objects (profile, recipe, run
manifest), content hashing, file io, and ``michi.toml`` resolution as the
milestones that need them land (see PLAN.md §8 and §15). Today it exposes
only the error hierarchy.

Design Principles
-----------------
- ``core`` depends on nothing else in michi; every domain module depends on
  ``core``. The dependency direction is one-way.
- Artifacts are immutable value objects with versioned schemas.
- No speculative infrastructure: helpers land here in the milestone where a
  real consumer first needs them, never before.
"""

from __future__ import annotations

from michi.core.errors import (
    DataError,
    MichiError,
    ModelError,
    RecipeError,
    ReportError,
    RunError,
)

__all__ = [
    "DataError",
    "MichiError",
    "ModelError",
    "RecipeError",
    "ReportError",
    "RunError",
]
