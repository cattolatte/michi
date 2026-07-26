"""Explanations attached to findings.

michi explains what an observation *means* and which options exist, never
which option to take. Text lives in ``content/`` as reviewable data files.

Design Principles
-----------------
- Explanations attach to observations, never to recommendations michi made,
  because michi does not make recommendations.
- Additive for learners, ignorable for experts: no output is blocked on them.
- Content is data, so accuracy fixes never touch program logic.
"""

from __future__ import annotations

from michi.explain.registry import Explanation, explanation_for, explanations

__all__ = ["Explanation", "explanation_for", "explanations"]
