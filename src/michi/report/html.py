"""Rendering artifacts to self-contained HTML.

Design Principles
-----------------
- **One file, no network.** Reports embed their own CSS and draw charts as
  inline SVG. No CDN, no bundler, no JavaScript dependency — a michi report
  opens on a locked-down laptop, from an email attachment, in five years.
- **Charts come from the artifact.** Histograms are binned during profiling
  and stored, so HTML, terminal, and any future viewer draw identical
  distributions.
- Templates are content: presentation lives in ``templates/``, not in Python
  string concatenation.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import TYPE_CHECKING

from michi.core.artifacts import ColumnKind, ColumnProfile, DatasetProfile
from michi.explain import explanation_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jinja2 import Environment

__all__ = ["render_profile_html"]

_SPARK_WIDTH = 104
_SPARK_HEIGHT = 26


@lru_cache(maxsize=1)
def _environment() -> Environment:
    """Build the Jinja environment, autoescaping HTML by default."""
    from jinja2 import Environment, PackageLoader, select_autoescape

    return Environment(
        loader=PackageLoader("michi.report", "templates"),
        autoescape=select_autoescape(default_for_string=True, default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_profile_html(profile: DatasetProfile) -> str:
    """Render a dataset profile as a single self-contained HTML document.

    Parameters
    ----------
    profile
        The artifact to render.

    Returns
    -------
    str
        Complete HTML, safe to write to disk and open offline.
    """
    findings = profile.findings_by_severity()

    seen: list[str] = []
    for finding in findings:
        if finding.kind not in seen:
            seen.append(finding.kind)
    explanations = [
        explanation
        for explanation in (explanation_for(kind) for kind in seen)
        if explanation is not None
    ]

    template = _environment().get_template("profile.html.jinja")
    return template.render(
        profile=profile,
        findings=findings,
        explanations=explanations,
        file_name=profile.source.path,
        sparkline=_sparkline,
        summarise=_summarise,
    )


def _sparkline(column: ColumnProfile) -> str:
    """Return an inline SVG distribution chart for a column."""
    if column.histogram:
        return _histogram_svg(column)
    if column.top_values:
        return _category_svg(column)
    return '<span class="muted">—</span>'


def _histogram_svg(column: ColumnProfile) -> str:
    counts = [count for _, _, count in column.histogram]
    peak = max(counts) if counts else 0
    if peak <= 0:
        return '<span class="muted">—</span>'

    bars = len(counts)
    slot = _SPARK_WIDTH / bars
    width = max(1.0, slot - 1.0)
    parts = [
        f'<svg class="spark" width="{_SPARK_WIDTH}" height="{_SPARK_HEIGHT}" '
        f'viewBox="0 0 {_SPARK_WIDTH} {_SPARK_HEIGHT}" role="img" '
        f'aria-label="distribution">'
    ]
    for index, count in enumerate(counts):
        height = max(1.0, (count / peak) * (_SPARK_HEIGHT - 2))
        x = index * slot
        y = _SPARK_HEIGHT - height
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
            f'height="{height:.2f}" fill="var(--bar-strong)" rx="0.5"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _category_svg(column: ColumnProfile) -> str:
    counts = [count for _, count in column.top_values]
    peak = max(counts) if counts else 0
    if peak <= 0:
        return '<span class="muted">—</span>'

    rows = len(counts)
    row_height = _SPARK_HEIGHT / rows
    height = max(3.0, row_height - 1.5)
    parts = [
        f'<svg class="spark" width="{_SPARK_WIDTH}" height="{_SPARK_HEIGHT}" '
        f'viewBox="0 0 {_SPARK_WIDTH} {_SPARK_HEIGHT}" role="img" '
        f'aria-label="top categories">'
    ]
    for index, count in enumerate(counts):
        bar = max(1.0, (count / peak) * _SPARK_WIDTH)
        y = index * row_height
        parts.append(
            f'<rect x="0" y="{y:.2f}" width="{bar:.2f}" height="{height:.2f}" '
            f'fill="var(--bar)" rx="1"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _summarise(column: ColumnProfile) -> str:
    """One-line human summary of a column, mirroring the terminal renderer."""
    if column.kind is ColumnKind.NUMERIC and column.stats:
        stats = column.stats
        parts = [
            f"mean {_number(stats.get('mean'))}",
            f"range {_number(stats.get('min'))}–{_number(stats.get('max'))}",
        ]
        skew = stats.get("skew")
        if skew is not None and abs(skew) >= 1.0:
            parts.append(f"skew {skew:+.2f}")
        outliers = stats.get("outliers")
        if outliers:
            parts.append(f"{int(outliers)} outliers")
        return " · ".join(parts)

    if column.kind is ColumnKind.DATETIME and column.stats:
        start = _epoch_to_date(column.stats.get("min_epoch_s"))
        end = _epoch_to_date(column.stats.get("max_epoch_s"))
        return f"{start} → {end}"

    if column.kind is ColumnKind.TEXT and column.stats:
        return (
            f"length {int(column.stats.get('min_length', 0))}–"
            f"{int(column.stats.get('max_length', 0))}"
        )

    if column.top_values:
        return "top: " + ", ".join(
            f"{value} ({count:,})" for value, count in column.top_values[:3]
        )
    return ""


def _number(value: float | None) -> str:
    if value is None or value != value:
        return "—"
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e6 or magnitude < 1e-3):
        return f"{value:.3g}"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _epoch_to_date(epoch: float | None) -> str:
    if epoch is None:
        return "—"
    try:
        return _dt.datetime.fromtimestamp(epoch, tz=_dt.UTC).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return "—"
