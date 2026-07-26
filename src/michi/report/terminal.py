"""Rendering artifacts to the terminal.

Design Principles
-----------------
- **The terminal is the primary surface.** HTML and JSON are alternatives, not
  the real output; anything a user needs must be legible here first.
- **Density with hierarchy.** An expert should be able to read the whole state
  of a dataset in one screen, with severity carrying the eye.
- **Explanations are opt-in.** Findings always render; the prose behind them
  appears only with ``--explain``, so the default output never lectures.
- Rendering never computes: everything shown comes from the artifact, so the
  terminal and HTML renderers can never disagree.
"""

from __future__ import annotations

import datetime as _dt

from rich import box
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from michi.core.artifacts import (
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    Finding,
    Severity,
)
from michi.core.manifest import RunManifest
from michi.explain import explanation_for

__all__ = ["render_evaluation", "render_profile", "severity_style"]

_KIND_STYLE = {
    ColumnKind.NUMERIC: "cyan",
    ColumnKind.CATEGORICAL: "magenta",
    ColumnKind.BOOLEAN: "green",
    ColumnKind.DATETIME: "blue",
    ColumnKind.TEXT: "yellow",
    ColumnKind.EMPTY: "dim",
}

_SEVERITY_STYLE = {
    Severity.HIGH: "bold red",
    Severity.WARN: "yellow",
    Severity.INFO: "dim",
}


def severity_style(severity: Severity) -> str:
    """Return the rich style used for a severity level.

    Examples
    --------
    >>> severity_style(Severity.HIGH)
    'bold red'
    """
    return _SEVERITY_STYLE[severity]


def render_profile(
    profile: DatasetProfile,
    console: Console,
    *,
    explain: bool = False,
    max_columns: int | None = None,
) -> None:
    """Render a dataset profile to the terminal.

    Parameters
    ----------
    profile
        The artifact to render.
    console
        Destination console.
    explain
        Also print what each finding kind means and which options exist.
    max_columns
        Truncate the column table after this many rows (``None`` shows all).
    """
    console.print()
    console.print(_header(profile))
    console.print()
    console.print(Padding(_summary(profile), (0, 0, 1, 2)))
    console.print(Padding(_columns_table(profile, max_columns), (0, 0, 1, 2)))

    findings = profile.findings_by_severity()
    if findings:
        console.print(Padding(_findings_table(findings), (0, 0, 1, 2)))
        if explain:
            console.print(Padding(_explanations(findings), (0, 0, 1, 2)))
        elif any(f.severity is not Severity.INFO for f in findings):
            console.print(
                Padding(
                    Text(
                        "Run again with --explain for what each finding means "
                        "and your options.",
                        style="dim",
                    ),
                    (0, 0, 1, 2),
                )
            )
    else:
        console.print(
            Padding(
                Text("No findings — nothing stood out.", style="green"), (0, 0, 1, 2)
            )
        )


def _header(profile: DatasetProfile) -> RenderableType:
    name = profile.source.path.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    text = Text()
    text.append(" 道 ", style="bold red")
    text.append(" michi inspect", style="bold")
    text.append("  ·  ", style="dim")
    text.append(name, style="bold cyan")
    return text


def _summary(profile: DatasetProfile) -> RenderableType:
    source = profile.source
    line = Text()
    line.append(f"{profile.n_rows:,} rows", style="bold")
    line.append(" × ", style="dim")
    line.append(f"{profile.n_columns:,} columns", style="bold")
    line.append("  ·  ", style="dim")
    line.append(f"{profile.missing_pct:.1f}% of cells missing")
    line.append("  ·  ", style="dim")
    line.append(f"{profile.duplicate_rows:,} duplicate rows")
    if profile.target:
        line.append("  ·  ", style="dim")
        line.append("target ", style="dim")
        line.append(profile.target, style="bold green")

    lines: list[RenderableType] = [line]
    if source.sampled:
        note = Text(style="yellow")
        note.append(
            f"sampled {source.sample_rows:,} of {source.total_rows:,} rows "
            f"(seed {source.seed}) — pass --full to read everything"
        )
        lines.append(note)
    provenance = Text(style="dim")
    provenance.append(
        f"sha256 {source.sha256[:12]}  ·  {_format_bytes(source.size_bytes)}"
    )
    lines.append(provenance)
    return Group(*lines)


def _columns_table(profile: DatasetProfile, max_columns: int | None) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold dim",
        pad_edge=False,
        show_edge=False,
        expand=False,
    )
    table.add_column("column", overflow="fold")
    table.add_column("kind")
    table.add_column("missing", justify="right")
    table.add_column("unique", justify="right")
    table.add_column("summary", overflow="fold")

    columns = profile.columns
    truncated = 0
    if max_columns is not None and len(columns) > max_columns:
        truncated = len(columns) - max_columns
        columns = columns[:max_columns]

    for column in columns:
        name = Text(column.name)
        if profile.target and column.name == profile.target:
            name.stylize("bold green")
        missing = Text(
            "—" if not column.missing else f"{column.missing_pct:.1f}%",
            style=_missing_style(column.missing_pct),
        )
        table.add_row(
            name,
            Text(column.kind.value, style=_KIND_STYLE[column.kind]),
            missing,
            f"{column.unique:,}",
            Text(_column_summary(column), style="dim"),
        )

    if truncated:
        table.add_row(
            Text(f"… {truncated} more columns", style="dim italic"), "", "", "", ""
        )
    return table


def _missing_style(pct: float) -> str:
    if pct >= 50.0:
        return "bold red"
    if pct >= 5.0:
        return "yellow"
    if pct > 0:
        return "dim"
    return "dim"


def _column_summary(column: ColumnProfile) -> str:
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
        return "  ·  ".join(parts)

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
        shown = ", ".join(
            f"{value} ({count:,})" for value, count in column.top_values[:3]
        )
        return f"top: {shown}"

    return ""


def _findings_table(findings: tuple[Finding, ...]) -> RenderableType:
    heading = Text()
    heading.append("Findings", style="bold")
    heading.append(f" ({len(findings)})", style="dim")

    table = Table(
        box=None, show_header=False, pad_edge=False, show_edge=False, padding=(0, 1)
    )
    table.add_column("severity", width=6)
    table.add_column("where", overflow="fold", max_width=28)
    table.add_column("what", overflow="fold")

    for finding in findings:
        where = ", ".join(finding.columns) if finding.columns else "dataset"
        table.add_row(
            Text(finding.severity.value, style=severity_style(finding.severity)),
            Text(where, style="bold"),
            Text(finding.summary),
        )
    return Group(heading, Text(), table)


def _explanations(findings: tuple[Finding, ...]) -> RenderableType:
    seen: list[str] = []
    for finding in findings:
        if finding.kind not in seen:
            seen.append(finding.kind)

    blocks: list[RenderableType] = [Text("What these mean", style="bold"), Text()]
    for kind in seen:
        explanation = explanation_for(kind)
        if explanation is None:
            continue
        title = Text()
        title.append(explanation.title, style="bold")
        title.append(f"  ({kind})", style="dim")
        blocks.append(title)
        blocks.append(Padding(Text(explanation.what), (0, 0, 0, 2)))
        if explanation.options:
            blocks.append(Padding(Text("Options:", style="dim"), (0, 0, 0, 2)))
            for option in explanation.options:
                bullet = Text()
                bullet.append("· ", style="dim")
                bullet.append(option)
                blocks.append(Padding(bullet, (0, 0, 0, 4)))
        if explanation.caution:
            caution = Text()
            caution.append("Caution: ", style="bold yellow")
            caution.append(explanation.caution)
            blocks.append(Padding(caution, (0, 0, 0, 2)))
        blocks.append(Text())
    return Group(*blocks)


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    if value != value:  # NaN
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


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def render_evaluation(
    manifest: RunManifest,
    console: Console,
    *,
    explain: bool = False,
) -> None:
    """Render a run manifest from ``michi eval`` to the terminal.

    Parameters
    ----------
    manifest
        The artifact to render.
    console
        Destination console.
    explain
        Also print what each check means and which options exist.
    """
    console.print()
    console.print(_eval_header(manifest))
    console.print()
    console.print(Padding(_eval_summary(manifest), (0, 0, 1, 2)))
    console.print(Padding(_metrics_table(manifest), (0, 0, 1, 2)))

    confusion = _confusion_table(manifest)
    if confusion is not None:
        console.print(Padding(confusion, (0, 0, 1, 2)))

    slices = _slices_table(manifest)
    if slices is not None:
        console.print(Padding(slices, (0, 0, 1, 2)))

    if manifest.checks:
        console.print(Padding(_findings_table(manifest.checks), (0, 0, 1, 2)))
        if explain:
            console.print(Padding(_explanations(manifest.checks), (0, 0, 1, 2)))
        else:
            console.print(
                Padding(
                    Text(
                        "Run again with --explain for what each check means "
                        "and your options.",
                        style="dim",
                    ),
                    (0, 0, 1, 2),
                )
            )


def _eval_header(manifest: RunManifest) -> RenderableType:
    text = Text()
    text.append(" 道 ", style="bold red")
    text.append(" michi eval", style="bold")
    text.append("  ·  ", style="dim")
    text.append(manifest.model.class_name, style="bold cyan")
    return text


def _eval_summary(manifest: RunManifest) -> RenderableType:
    name = manifest.dataset.path.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    line = Text()
    line.append(manifest.task, style="bold")
    line.append("  ·  ", style="dim")
    line.append(f"{manifest.n_rows:,} rows", style="bold")
    line.append("  ·  ", style="dim")
    line.append(name)
    line.append("  ·  ", style="dim")
    line.append("target ", style="dim")
    line.append(manifest.target, style="bold green")

    provenance = Text(style="dim")
    provenance.append(
        f"run {manifest.run_id}  ·  data {manifest.dataset.sha256[:12]}  ·  "
        f"seed {manifest.seed}  ·  {manifest.duration_s:.2f}s"
    )
    return Group(line, provenance)


def _metrics_table(manifest: RunManifest) -> RenderableType:
    heading = Text("Metrics", style="bold")
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold dim",
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("metric", no_wrap=True)
    table.add_column("value", justify="right", no_wrap=True)
    table.add_column("95% interval", justify="right", no_wrap=True)
    for name in sorted(manifest.baselines):
        table.add_column(f"vs {name}", justify="right", no_wrap=True)

    baseline_names = sorted(manifest.baselines)
    for index, metric in enumerate(manifest.metrics):
        interval = (
            f"{metric.ci_low:.4g} – {metric.ci_high:.4g}"
            if metric.has_interval
            else "—"
        )
        row = [
            Text(metric.name, style="bold" if index == 0 else ""),
            Text(_number(metric.value), style="bold" if index == 0 else ""),
            Text(interval, style="dim"),
        ]
        for baseline in baseline_names:
            row.append(
                Text(
                    _baseline_value(manifest, baseline, metric.name),
                    style="dim",
                )
            )
        table.add_row(*row)
    return Group(heading, Text(), table)


def _baseline_value(manifest: RunManifest, baseline: str, metric_name: str) -> str:
    for metric in manifest.baselines.get(baseline, ()):
        if metric.name == metric_name:
            return _number(metric.value)
    return "—"


def _confusion_table(manifest: RunManifest) -> RenderableType | None:
    classes = manifest.details.get("classes")
    confusion = manifest.details.get("confusion")
    if not classes or not confusion or len(classes) > 12:
        return None

    heading = Text("Confusion", style="bold")
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold dim",
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("actual ╲ predicted", no_wrap=True)
    for label in classes:
        table.add_column(str(label), justify="right")

    for index, label in enumerate(classes):
        cells = [Text(str(label), style="bold")]
        for column, count in enumerate(confusion[index]):
            style = "green" if column == index else ("red" if count else "dim")
            cells.append(Text(f"{count:,}", style=style))
        table.add_row(*cells)
    return Group(heading, Text(), table)


def _slices_table(manifest: RunManifest) -> RenderableType | None:
    slices = manifest.details.get("slices") or []
    if not slices:
        return None

    ranked = sorted(slices, key=lambda item: item["score"])[:8]
    heading = Text()
    heading.append("Slices", style="bold")
    heading.append(f"  ({len(slices)} subgroups, weakest first)", style="dim")

    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold dim",
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("column")
    table.add_column("value")
    table.add_column("rows", justify="right")
    table.add_column("score", justify="right")
    for item in ranked:
        table.add_row(
            Text(str(item["column"])),
            Text(str(item["value"]), style="bold"),
            f"{int(item['n_rows']):,}",
            Text(_number(float(item["score"])), style="dim"),
        )
    return Group(heading, Text(), table)
