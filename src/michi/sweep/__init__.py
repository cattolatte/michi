"""Experiment grids — the ``michi sweep`` verb.

Executes a declarative grid of models × recipes × seeds, recording one run
manifest per cell, with content-hash caching so an interrupted sweep resumes
exactly where it stopped.

Design Principles
-----------------
- The grid is a file you can read, diff, and put in a paper's appendix.
- Caching is by content: a cell is reused only when data, recipe, model, seed,
  and folds all hash identically.
- One cell's failure never costs the cells already completed.
"""

from __future__ import annotations

from michi.sweep.plan import SWEEP_SCHEMA_VERSION, SweepCell, SweepPlan, load_plan
from michi.sweep.runner import CellOutcome, SweepResult, run_sweep

__all__ = [
    "SWEEP_SCHEMA_VERSION",
    "CellOutcome",
    "SweepCell",
    "SweepPlan",
    "SweepResult",
    "load_plan",
    "run_sweep",
]
