"""Rendering michi artifacts for humans.

Terminal output is the primary surface; HTML is a self-contained offline
alternative. Markdown and LaTeX renderers arrive with ``michi report`` in
v0.3.

Design Principles
-----------------
- Renderers read artifacts and nothing else, so two surfaces can never
  disagree about what the data says.
- HTML reports are single files with no network dependency of any kind.
- Presentation lives in templates and stylesheets, not in Python string
  concatenation.
"""

from __future__ import annotations

from michi.report.html import render_profile_html
from michi.report.terminal import render_evaluation, render_profile

__all__ = ["render_evaluation", "render_profile", "render_profile_html"]
