"""Model evaluation — the ``michi eval`` verb.

Evaluates a model michi never saw trained: metrics with confidence intervals,
trivial baselines for comparison, calibration, per-slice performance, and
checks that catch the mistakes reviewers actually look for.

The package is named ``evaluation`` (not ``eval``) to avoid shadowing the
built-in; the CLI verb remains ``michi eval``.

Design Principles
-----------------
- The model is a black box with ``predict``; nothing about how it was built
  is assumed or required.
- Baselines and uncertainty are always computed, never opt-in, because a score
  without either invites conclusions the data cannot support.
- Results are recorded as run manifests, so every number is traceable to the
  bytes, seed, and environment that produced it.
"""

from __future__ import annotations

from michi.evaluation.checks import evaluation_checks
from michi.evaluation.evaluator import SliceMetric, evaluate_model, new_run_id
from michi.evaluation.metrics import (
    classification_metrics,
    detect_task,
    regression_metrics,
)

__all__ = [
    "SliceMetric",
    "classification_metrics",
    "detect_task",
    "evaluate_model",
    "evaluation_checks",
    "new_run_id",
    "regression_metrics",
]
