"""Core: shared foundations for every michi verb.

This package holds the artifact value objects (profile, recipe, run manifest),
content hashing, tabular io, and the error hierarchy. Domain packages build on
it; it depends on nothing else in michi.

Design Principles
-----------------
- ``core`` depends on nothing else in michi; every domain module depends on
  ``core``. The dependency direction is one-way and enforced by review.
- Artifacts are immutable value objects with explicitly versioned schemas.
- No speculative infrastructure: helpers land here in the milestone where a
  real consumer first needs them, never before.
"""

from __future__ import annotations

from michi.core.artifacts import (
    PROFILE_SCHEMA_VERSION,
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    Finding,
    Severity,
    SourceInfo,
    utc_now_iso,
)
from michi.core.errors import (
    DataError,
    MichiError,
    ModelError,
    RecipeError,
    ReportError,
    RunError,
)
from michi.core.hashing import hash_file, hash_payload
from michi.core.io import LoadedTable, load_table, supported_formats

__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "ColumnKind",
    "ColumnProfile",
    "DataError",
    "DatasetProfile",
    "Finding",
    "LoadedTable",
    "MichiError",
    "ModelError",
    "RecipeError",
    "ReportError",
    "RunError",
    "Severity",
    "SourceInfo",
    "hash_file",
    "hash_payload",
    "load_table",
    "supported_formats",
    "utc_now_iso",
]
