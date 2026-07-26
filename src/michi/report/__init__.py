"""Rendering michi artifacts for humans.

Terminal output is the primary surface; HTML, Markdown, and LaTeX are
alternatives over the same artifacts.

Design Principles
-----------------
- Renderers read artifacts and nothing else, so two surfaces can never
  disagree about what the data says.
- HTML reports are single files with no network dependency of any kind.
- Presentation lives in templates and stylesheets, not in Python string
  concatenation.
- Statistical honesty travels with the numbers into every format, including
  the LaTeX table destined for a paper.
"""

from __future__ import annotations

from michi.report.comparison import (
    render_benchmark_html,
    render_benchmark_latex,
    render_benchmark_markdown,
    render_runs_html,
    render_runs_latex,
    render_runs_markdown,
)
from michi.report.html import render_profile_html
from michi.report.runs import RunGroup, group_runs, load_manifests
from michi.report.terminal import (
    render_benchmark,
    render_evaluation,
    render_profile,
    render_runs_terminal,
)

__all__ = [
    "RunGroup",
    "group_runs",
    "load_manifests",
    "render_benchmark",
    "render_benchmark_html",
    "render_benchmark_latex",
    "render_benchmark_markdown",
    "render_evaluation",
    "render_profile",
    "render_profile_html",
    "render_runs_html",
    "render_runs_latex",
    "render_runs_markdown",
    "render_runs_terminal",
]
