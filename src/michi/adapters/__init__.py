"""Model loading adapters.

Design Principles
-----------------
- Two ways in: sklearn-compatible pickles, and a ``module:object`` predict
  protocol that covers every other framework without per-framework loaders.
- Formats that cannot be loaded honestly are refused with an explanation,
  never guessed at.
- The model stays a black box: michi calls ``predict`` and never inspects
  internals.
"""

from __future__ import annotations

from michi.adapters.model import LoadedModel, load_model

__all__ = ["LoadedModel", "load_model"]
