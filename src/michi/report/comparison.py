"""Rendering benchmark comparisons to files.

Design Principles
-----------------
- **Every format says the same thing.** HTML, Markdown, and LaTeX are
  renderings of one artifact set, so a paper table and a browser page can
  never disagree about which model won.
- **The honesty travels with the numbers.** Confidence intervals and the
  "not distinguishable from the leader" verdict appear in every format,
  including the LaTeX table meant for a paper.
- **LaTeX output is paste-ready.** A researcher should be able to drop the
  table into a manuscript unedited, which means booktabs, aligned decimals,
  and a caption that states the test used.
- HTML remains a single offline file: no CDN, no JavaScript, no build step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from michi.core.manifest import Metric, RunManifest
from michi.report.runs import RunGroup
from michi.report.terminal import short_run_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from michi.bench import BenchResult

__all__ = [
    "render_benchmark_html",
    "render_benchmark_latex",
    "render_benchmark_markdown",
    "render_runs_html",
    "render_runs_latex",
    "render_runs_markdown",
]


def render_benchmark_html(result: BenchResult) -> str:
    """Render a benchmark as a single self-contained HTML document."""
    from michi.bench import describe_policy
    from michi.report.html import _environment

    template = _environment().get_template("benchmark.html.jinja")
    return template.render(
        result=result,
        rows=_rows(result),
        preparation=describe_policy(result.policy, scaled=True),
        tied=_tied(result),
    )


def render_benchmark_markdown(result: BenchResult) -> str:
    """Render a benchmark as Markdown, for a README or a pull request."""
    from michi.bench import describe_policy

    lines = [
        f"# Benchmark — {result.target}",
        "",
        f"- **Task**: {result.task}",
        f"- **Rows**: {result.n_rows:,}",
        f"- **Cross-validation**: {result.folds}-fold, seed {result.seed}",
        f"- **Preparation**: {describe_policy(result.policy, scaled=True)}",
        "",
        f"| Model | {result.primary_metric} | 95% interval | vs leader |",
        "|---|---:|---:|---|",
    ]
    for row in _rows(result):
        lines.append(
            f"| {row['name']} | {row['value']} | {row['interval']} | {row['verdict']} |"
        )

    lines.extend(["", "## Verdict", "", _verdict_sentence(result), ""])
    if result.checks:
        lines.extend(["## Checks", ""])
        lines.extend(
            f"- **{check.severity.value}** — {check.summary}" for check in result.checks
        )
        lines.append("")
    lines.append(
        "_Differences tested with the corrected resampled t-test "
        "(Nadeau & Bengio, 2003), Holm-adjusted across models._"
    )
    return "\n".join(lines) + "\n"


def render_benchmark_latex(result: BenchResult) -> str:
    """Render a benchmark as a booktabs LaTeX table, ready to paste."""
    metric = _escape_latex(result.primary_metric.replace("_", " "))
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Cross-validated {metric} on \\texttt{{"
        f"{_escape_latex(result.target)}}} "
        f"({result.folds}-fold, {result.n_rows:,} rows, seed {result.seed}). "
        "Intervals are 95\\% across folds; differences from the leading model "
        "are tested with the corrected resampled $t$-test "
        "\\citep{nadeau2003inference} and Holm-adjusted.}",
        "\\label{tab:michi-benchmark}",
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        f"Model & {metric} & 95\\% CI & vs.\\ leader \\\\",
        "\\midrule",
    ]
    for row in _rows(result):
        name = _escape_latex(str(row["name"]))
        if row["is_leader"]:
            name = f"\\textbf{{{name}}}"
        lines.append(
            f"{name} & {row['value']} & {_escape_latex(str(row['interval']))} "
            f"& {_escape_latex(str(row['verdict']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _rows(result: BenchResult) -> list[dict[str, object]]:
    """Flatten a benchmark into rows every renderer can share."""
    verdicts = {item.model: item for item in result.comparisons}
    rows: list[dict[str, object]] = []
    for item in result.results:
        if item.failed is not None:
            rows.append(
                {
                    "name": item.name,
                    "value": "failed",
                    "interval": "—",
                    "verdict": item.failed,
                    "is_leader": False,
                    "failed": True,
                }
            )
            continue

        comparison = verdicts.get(item.name)
        is_leader = comparison is not None and comparison.model == comparison.leader
        if comparison is None:
            verdict = "—"
        elif is_leader:
            verdict = "leader"
        elif comparison.significant:
            verdict = f"worse ({comparison.formatted_p})"
        else:
            verdict = f"tied with leader ({comparison.formatted_p})"

        rows.append(
            {
                "name": item.name,
                "value": _format_metric(item.primary),
                "interval": _format_interval(item.primary),
                "verdict": verdict,
                "is_leader": is_leader,
                "failed": False,
            }
        )
    return rows


def _tied(result: BenchResult) -> list[str]:
    """Names of models the data cannot separate from the leader."""
    return [
        comparison.model
        for comparison in result.comparisons
        if comparison.model != comparison.leader and not comparison.significant
    ]


def _verdict_sentence(result: BenchResult) -> str:
    """The plain-language conclusion, shared by every file format."""
    leader = result.leader
    if leader is None:
        return "No model could be trained."
    tied = _tied(result)
    if not tied:
        return (
            f"`{leader.name}` scores highest, and its advantage over every "
            f"other model is statistically significant."
        )
    names = ", ".join(f"`{name}`" for name in tied)
    verb = "is" if len(tied) == 1 else "are"
    return (
        f"`{leader.name}` scores highest, but {names} {verb} statistically "
        f"indistinguishable from it at this sample size. Choosing between "
        f"them on these numbers alone is not supported."
    )


def _format_metric(metric: Metric) -> str:
    return f"{metric.value:.4g}"


def _format_interval(metric: Metric) -> str:
    if not metric.has_interval:
        return "—"
    return f"{metric.ci_low:.4g}–{metric.ci_high:.4g}"


def _escape_latex(text: str) -> str:
    """Escape the characters LaTeX would otherwise interpret."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "—": "---",
        "–": "--",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_runs_markdown(groups: tuple[RunGroup, ...]) -> str:
    """Render recorded runs as Markdown."""
    lines = ["# michi runs", ""]
    for group in groups:
        lines.extend(
            [
                f"## {group.dataset} — {group.target}",
                "",
                f"- **Task**: {group.task}",
                f"- **Runs**: {len(group.manifests)}",
                f"- **Data**: `{group.dataset_sha[:16]}`",
                "",
                f"| Run | Model | {group.primary_metric} | 95% interval | verdict |",
                "|---|---|---:|---:|---|",
            ]
        )
        for manifest in group.ranked():
            metric = manifest.primary if manifest.metrics else None
            verdict = _manifest_verdict(manifest)
            lines.append(
                f"| `{short_run_id(manifest.run_id)}` | {manifest.model.class_name} | "
                f"{_format_metric(metric) if metric else '—'} | "
                f"{_format_interval(metric) if metric else '—'} | {verdict} |"
            )
        lines.append("")
    lines.append(
        "_Runs are grouped by dataset hash and target: only runs within a "
        "group are comparable._"
    )
    return "\n".join(lines) + "\n"


def render_runs_latex(groups: tuple[RunGroup, ...]) -> str:
    """Render recorded runs as booktabs LaTeX tables."""
    blocks: list[str] = ["% Requires \\usepackage{booktabs}"]
    for index, group in enumerate(groups):
        metric = _escape_latex(group.primary_metric.replace("_", " "))
        blocks.extend(
            [
                "\\begin{table}[t]",
                "\\centering",
                f"\\caption{{Recorded runs on \\texttt{{"
                f"{_escape_latex(group.dataset)}}}, target \\texttt{{"
                f"{_escape_latex(group.target)}}}. Intervals are 95\\%.}}",
                f"\\label{{tab:michi-runs-{index}}}",
                "\\begin{tabular}{llrr}",
                "\\toprule",
                f"Run & Model & {metric} & 95\\% CI \\\\",
                "\\midrule",
            ]
        )
        for manifest in group.ranked():
            item = manifest.primary if manifest.metrics else None
            blocks.append(
                f"\\texttt{{{_escape_latex(short_run_id(manifest.run_id))}}} & "
                f"{_escape_latex(manifest.model.class_name)} & "
                f"{_format_metric(item) if item else '--'} & "
                f"{_escape_latex(_format_interval(item)) if item else '--'} \\\\"
            )
        blocks.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(blocks)


def render_runs_html(groups: tuple[RunGroup, ...]) -> str:
    """Render recorded runs as a single self-contained HTML document."""
    from michi.report.html import _environment

    template = _environment().get_template("runs.html.jinja")
    return template.render(
        groups=[
            {
                "group": group,
                "rows": [
                    {
                        "run_id": manifest.run_id,
                        "short_id": short_run_id(manifest.run_id),
                        "model": manifest.model.class_name,
                        "reference": manifest.model.reference,
                        "kind": manifest.kind,
                        "value": (
                            _format_metric(manifest.primary)
                            if manifest.metrics
                            else "—"
                        ),
                        "interval": (
                            _format_interval(manifest.primary)
                            if manifest.metrics
                            else "—"
                        ),
                        "verdict": _manifest_verdict(manifest),
                        "created_at": manifest.created_at,
                        "checks": manifest.checks,
                    }
                    for manifest in group.ranked()
                ],
            }
            for group in groups
        ],
        total=sum(len(group.manifests) for group in groups),
    )


def _manifest_verdict(manifest: RunManifest) -> str:
    """The comparison verdict a manifest recorded, if it recorded one."""
    comparison = manifest.details.get("comparison")
    if not isinstance(comparison, dict):
        return "—"
    if comparison.get("model") == comparison.get("leader"):
        return "leader"
    if comparison.get("significant"):
        return f"worse (p={float(comparison.get('adjusted_p', 1.0)):.3g})"
    return f"tied with leader (p={float(comparison.get('adjusted_p', 1.0)):.3g})"
