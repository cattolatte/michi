"""Dataset profiling — the ``michi inspect`` verb.

Profiles a tabular file and reports what is true about it: column kinds,
missing values, duplicates, cardinality, distribution shape, outliers,
redundancy, and — when a target is named — class imbalance and leakage
suspects.

The package is named ``inspection`` (not ``inspect``) to avoid shadowing the
standard-library module; the CLI verb remains ``michi inspect``.

Design Principles
-----------------
- Describe, never prescribe: findings state facts; options come from the
  explanation layer; the decision is the user's.
- Thresholds are explicit, named, and documented rather than hidden.
- Profiling a large file must stay fast, so sampling is automatic and always
  recorded in the artifact.
"""

from __future__ import annotations

from michi.inspection.findings import Thresholds, detect_findings
from michi.inspection.profiler import profile_table

__all__ = ["Thresholds", "detect_findings", "profile_table"]
