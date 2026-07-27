"""Inline SVG charts drawn from artifacts michi has already recorded.

Design Principles
-----------------
- **Every chart is a rendering, never a computation.** The confusion counts,
  the calibration bins, the per-slice scores, and the intervals all come from
  the run manifest. A chart that recomputed anything could disagree with the
  terminal output, and then two of michi's surfaces would be telling the user
  different things about the same run.
- **SVG in the document, nothing fetched.** No CDN, no JavaScript, no plotting
  library at render time. The viewer works air-gapped, the HTML report is one
  file you can email, and neither can rot when a frontend toolchain moves on.
- **A chart says what it cannot show.** Too many classes for a legible
  confusion matrix, or too few bins for a calibration curve, returns ``None``
  so the caller can print the honest table instead of a misleading picture.
- **Colour carries meaning or nothing.** Severity and direction are coloured;
  decoration is not, because a reader who cannot distinguish two hues should
  lose nothing.
"""

from __future__ import annotations

from html import escape
from typing import Any

__all__ = [
    "calibration_chart",
    "confusion_chart",
    "importance_chart",
    "interval_chart",
    "slice_chart",
]

# Charts render into two documents with opposite backgrounds: the light HTML
# report and the dark local viewer. Text and rules therefore inherit the
# page's own colour instead of naming one — a hardcoded near-black is
# invisible on dark, which is exactly how these first shipped.
INK = "currentColor"
MUTED = "currentColor"
RULE = "currentColor"
GRID = "currentColor"

# Meaning-bearing colours are stated, because they must survive either
# background and both of these do.
ACCENT = "#c4453a"
GOOD = "#5a9e60"

_MUTED_OPACITY = "0.62"
_RULE_OPACITY = "0.30"
_GRID_OPACITY = "0.12"


def confusion_chart(
    classes: list[Any], confusion: list[list[int]], *, size: int = 260
) -> str | None:
    """A confusion matrix as a shaded grid.

    Rows are the true class, columns the predicted one, so the diagonal is
    what the model got right. Cells are shaded by their share of the *row*,
    not of the whole matrix: on an imbalanced problem a whole-matrix shading
    makes the majority class the only visible thing, which hides exactly the
    failure a confusion matrix is read to find.
    """
    if not classes or not confusion or len(classes) > 10:
        return None
    if len(confusion) != len(classes):
        return None

    count = len(classes)
    label_space = 74
    cell = max((size - label_space) // count, 26)
    width = label_space + cell * count + 16
    height = label_space + cell * count + 16

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="confusion matrix">'
    ]

    for row in range(count):
        total = sum(confusion[row]) or 1
        for column in range(count):
            value = int(confusion[row][column])
            share = value / total
            x = label_space + column * cell
            y = label_space + row * cell
            # The diagonal is success and the off-diagonal is error; giving
            # them the same colour would make a confident wrong answer look
            # like a confident right one.
            hue = GOOD if row == column else ACCENT
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                f'fill="{hue}" fill-opacity="{0.08 + 0.82 * share:.3f}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + (cell - 2) / 2:.1f}" y="{y + cell / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="11" '
                f'fill="{INK if share < 0.55 else "#ffffff"}">{value}</text>'
            )

    for index, name in enumerate(classes):
        label = escape(str(name))[:9]
        centre = label_space + index * cell + cell / 2
        parts.append(
            f'<text x="{label_space - 8}" y="{centre + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">'
            f"{label}</text>"
        )
        parts.append(
            f'<text x="{centre:.1f}" y="{label_space - 10}" text-anchor="middle" '
            f'font-size="11" fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">'
            f"{label}</text>"
        )

    parts.append(
        f'<text x="6" y="16" font-size="10" fill="{MUTED}" '
        f'fill-opacity="{_MUTED_OPACITY}">actual ↓ / predicted →</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def calibration_chart(
    calibration: list[Any] | None, *, width: int = 300, height: int = 220
) -> str | None:
    """A reliability diagram: predicted probability against observed rate.

    The diagonal is perfect calibration. A curve below it means the model is
    overconfident — it says 0.9 and is right 0.7 of the time — which matters
    whenever a probability is used as a probability rather than as a ranking.
    """
    if not calibration or len(calibration) < 2:
        return None

    points: list[tuple[float, float, int]] = []
    for entry in calibration:
        # Manifests record a bin as [predicted, observed, count]. A mapping is
        # accepted too so a hand-written or future artifact still draws.
        if isinstance(entry, dict):
            predicted = entry.get("predicted")
            observed = entry.get("observed")
            count = int(entry.get("count", 0))
        elif len(entry) >= 2:
            predicted, observed = entry[0], entry[1]
            count = int(entry[2]) if len(entry) > 2 else 0
        else:
            continue
        if predicted is None or observed is None:
            continue
        points.append((float(predicted), float(observed), count))
    if len(points) < 2:
        return None

    pad = 34
    plot_w = width - pad - 12
    plot_h = height - pad - 12

    def place(value: float) -> float:
        return max(0.0, min(1.0, value))

    def x_of(value: float) -> float:
        return pad + place(value) * plot_w

    def y_of(value: float) -> float:
        return 12 + (1 - place(value)) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="calibration curve">'
    ]
    for step in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{pad}" y1="{y_of(step):.1f}" x2="{pad + plot_w}" '
            f'y2="{y_of(step):.1f}" stroke="{GRID}" stroke-opacity="{_GRID_OPACITY}"/>'
        )
    parts.append(
        f'<line x1="{x_of(0)}" y1="{y_of(0):.1f}" x2="{x_of(1)}" y2="{y_of(1):.1f}" '
        f'stroke="{RULE}" stroke-opacity="{_RULE_OPACITY}" stroke-dasharray="4 3"/>'
    )

    path = " ".join(
        f"{'M' if index == 0 else 'L'}{x_of(p):.1f},{y_of(o):.1f}"
        for index, (p, o, _) in enumerate(sorted(points))
    )
    parts.append(f'<path d="{path}" fill="none" stroke="{ACCENT}" stroke-width="2"/>')
    for predicted, observed, _ in points:
        parts.append(
            f'<circle cx="{x_of(predicted):.1f}" cy="{y_of(observed):.1f}" r="3" '
            f'fill="{ACCENT}"/>'
        )

    parts.append(
        f'<text x="{pad}" y="{height - 6}" font-size="10" fill="{MUTED}" '
        f'fill-opacity="{_MUTED_OPACITY}">predicted probability →</text>'
    )
    parts.append(
        f'<text x="6" y="14" font-size="10" fill="{MUTED}" '
        f'fill-opacity="{_MUTED_OPACITY}">observed rate ↑</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def interval_chart(
    rows: list[tuple[str, float, float | None, float | None]],
    *,
    width: int = 420,
    greater_is_better: bool = True,
) -> str | None:
    """A forest plot of point estimates and their confidence intervals.

    Overlapping intervals are the visual form of the verdict michi already
    prints in words: two models whose bars overlap heavily are two models the
    data did not separate.
    """
    usable = [
        (name, value, low, high) for name, value, low, high in rows if value == value
    ]
    if not usable:
        return None

    values = [value for _, value, _, _ in usable]
    lows = [low for _, _, low, _ in usable if low is not None]
    highs = [high for _, _, _, high in usable if high is not None]
    smallest = min([*values, *lows])
    largest = max([*values, *highs])
    if largest == smallest:
        largest = smallest + 1e-9
    span = largest - smallest
    smallest -= span * 0.08
    largest += span * 0.08

    label_space = 132
    row_height = 26
    height = row_height * len(usable) + 26
    plot_w = width - label_space - 62

    def x_of(value: float) -> float:
        return label_space + (value - smallest) / (largest - smallest) * plot_w

    best = max(values) if greater_is_better else min(values)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="scores with confidence intervals">'
    ]
    for index, (name, value, low, high) in enumerate(usable):
        y = 16 + index * row_height
        leading = value == best
        colour = GOOD if leading else INK
        parts.append(
            f'<text x="{label_space - 10}" y="{y + 4}" text-anchor="end" '
            f'font-size="11" fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">'
            f"{escape(str(name))[:22]}</text>"
        )
        if low is not None and high is not None:
            parts.append(
                f'<line x1="{x_of(low):.1f}" y1="{y}" x2="{x_of(high):.1f}" y2="{y}" '
                f'stroke="{RULE}" stroke-opacity="{_RULE_OPACITY}" stroke-width="3"/>'
            )
        parts.append(f'<circle cx="{x_of(value):.1f}" cy="{y}" r="4" fill="{colour}"/>')
        parts.append(
            f'<text x="{width - 6}" y="{y + 4}" text-anchor="end" font-size="11" '
            f'fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">{value:.4g}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def slice_chart(
    slices: list[dict[str, Any]], *, width: int = 420, limit: int = 12
) -> str | None:
    """Per-subgroup scores as bars, worst first.

    A model's average hides the group it fails. Sorting worst-first puts the
    subgroup a reader most needs to see at the top, rather than requiring them
    to scan for it.
    """
    entries: list[tuple[str, float, int]] = []
    for entry in slices:
        # `value` is the slice's *label* in the manifest, not its score —
        # reading it as the score plots the group names as numbers.
        score = entry.get("score")
        if score is None:
            continue
        label = f"{entry.get('column', '?')} = {entry.get('value', '?')}"
        entries.append((label, float(score), int(entry.get("n_rows", 0))))
    if not entries:
        return None

    entries.sort(key=lambda item: item[1])
    entries = entries[:limit]

    label_space = 168
    row_height = 22
    height = row_height * len(entries) + 22
    plot_w = width - label_space - 96
    largest = max(value for _, value, _ in entries) or 1.0

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="performance by subgroup">'
    ]
    worst = entries[0][1]
    for index, (label, value, count) in enumerate(entries):
        y = 14 + index * row_height
        bar = max(value / largest * plot_w, 1.0)
        # The worst group is the point of the chart, so it is the one thing
        # coloured. Colouring every bar would make none of them stand out.
        colour = ACCENT if value == worst else MUTED
        parts.append(
            f'<text x="{label_space - 8}" y="{y + 4}" text-anchor="end" '
            f'font-size="10" fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">'
            f"{escape(label)[:30]}</text>"
        )
        parts.append(
            f'<rect x="{label_space}" y="{y - 6}" width="{bar:.1f}" height="12" '
            f'fill="{colour}" fill-opacity="0.75" rx="2"/>'
        )
        suffix = f" (n={count})" if count else ""
        parts.append(
            f'<text x="{width - 4}" y="{y + 4}" text-anchor="end" font-size="10" '
            f'fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">'
            f"{value:.3g}{escape(suffix)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def importance_chart(
    ranked: list[dict[str, Any]], *, width: int = 420, limit: int = 12
) -> str | None:
    """Columns ranked by what the model loses without them.

    Bars inside their own error bar are drawn faintly rather than omitted: a
    column measured as unimportant is a result, and hiding it would make the
    chart look more decisive than the run was.
    """
    entries: list[tuple[str, float, float]] = []
    for entry in ranked[:limit]:
        drop = entry.get("drop")
        if drop is None:
            continue
        entries.append(
            (
                str(entry.get("column", "?")),
                float(drop),
                float(entry.get("spread", 0.0)),
            )
        )
    if not entries:
        return None

    label_space = 150
    row_height = 22
    height = row_height * len(entries) + 22
    plot_w = width - label_space - 76
    largest = max(abs(value) for _, value, _ in entries) or 1.0

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="column importance">'
    ]
    for index, (label, drop, spread) in enumerate(entries):
        y = 14 + index * row_height
        bar = max(abs(drop) / largest * plot_w, 1.0)
        noise = abs(drop) <= spread
        parts.append(
            f'<text x="{label_space - 8}" y="{y + 4}" text-anchor="end" '
            f'font-size="10" fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">'
            f"{escape(label)[:24]}</text>"
        )
        parts.append(
            f'<rect x="{label_space}" y="{y - 6}" width="{bar:.1f}" height="12" '
            f'fill="{ACCENT if not noise else MUTED}" '
            f'fill-opacity="{0.75 if not noise else 0.25}" rx="2"/>'
        )
        suffix = "  within noise" if noise else ""
        parts.append(
            f'<text x="{width - 4}" y="{y + 4}" text-anchor="end" font-size="10" '
            f'fill="{MUTED}" fill-opacity="{_MUTED_OPACITY}">'
            f"{drop:+.3g}{escape(suffix)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)
