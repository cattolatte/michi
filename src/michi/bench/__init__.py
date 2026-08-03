"""Model benchmarking — the ``michi bench`` verb.

Trains the models you name under cross-validation and reports which of them
are actually distinguishable from one another, with a dummy baseline always
included as the floor.

Design Principles
-----------------
- The honest comparison is the product; training several models is the easy
  part.
- Preparation is fitted inside each fold, so a benchmark cannot leak.
- michi ranks and tests, but never picks the model for you.
"""

from __future__ import annotations

from michi.bench.preprocess import PreparationPolicy, describe_policy
from michi.bench.registry import ModelEntry, available_models, model_entry
from michi.bench.runner import (
    BenchResult,
    ModelResult,
    fold_pipeline,
    make_splitter,
    run_benchmark,
    scorers_for,
)
from michi.bench.significance import Comparison, corrected_paired_t_test

__all__ = [
    "BenchResult",
    "Comparison",
    "ModelEntry",
    "ModelResult",
    "PreparationPolicy",
    "available_models",
    "corrected_paired_t_test",
    "describe_policy",
    "fold_pipeline",
    "make_splitter",
    "model_entry",
    "run_benchmark",
    "scorers_for",
]
