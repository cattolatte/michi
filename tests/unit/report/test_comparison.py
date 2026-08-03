"""Paper-ready exports: the Markdown, LaTeX, and HTML a benchmark becomes.

These renderers are where a michi result stops being a terminal and starts
being something pasted into a README, a pull request, or a paper. The claim
they make — that the exported table says the same thing the terminal said — is
only true if the significance verdict, the interval, and the failed models all
survive the trip. This module was the least-covered in the package.
"""

from __future__ import annotations

import pytest

from michi.bench.preprocess import PreparationPolicy
from michi.bench.runner import BenchResult, ModelResult
from michi.bench.significance import Comparison
from michi.core.manifest import Metric
from michi.report.comparison import (
    render_benchmark_html,
    render_benchmark_latex,
    render_benchmark_markdown,
    render_runs_html,
    render_runs_latex,
    render_runs_markdown,
)

RENDERERS = (
    render_benchmark_markdown,
    render_benchmark_latex,
    render_benchmark_html,
)


def _model(
    name: str,
    value: float,
    *,
    metric: str = "balanced_accuracy",
    interval: tuple[float, float] | None = None,
    failed: str | None = None,
) -> ModelResult:
    low, high = interval or (value - 0.03, value + 0.03)
    return ModelResult(
        name=name,
        metrics=(
            Metric(
                name=metric,
                value=value,
                greater_is_better=metric != "rmse",
                ci_low=low,
                ci_high=high,
            ),
        ),
        fold_scores=(value, value, value, value, value),
        fit_seconds=0.2,
        failed=failed,
    )


def _comparison(model: str, leader: str, *, significant: bool, p: float) -> Comparison:
    return Comparison(
        model=model,
        leader=leader,
        difference=0.04,
        p_value=p,
        adjusted_p=p,
        significant=significant,
    )


@pytest.fixture
def result() -> BenchResult:
    """A leader, a model tied with it, and one that could not be trained."""
    return BenchResult(
        task="classification",
        target="purchased",
        folds=5,
        primary_metric="balanced_accuracy",
        results=(
            _model("gradient_boosting", 0.91),
            _model("linear", 0.87),
            _model("mlp", 0.0, failed="did not converge"),
        ),
        comparisons=(
            _comparison(
                "gradient_boosting", "gradient_boosting", significant=False, p=1.0
            ),
            _comparison("linear", "gradient_boosting", significant=False, p=0.31),
        ),
        policy=PreparationPolicy(),
        n_rows=600,
    )


# --- what every format must carry ------------------------------------------


@pytest.mark.parametrize("render", RENDERERS)
def test_every_format_names_every_model(result: BenchResult, render: object) -> None:
    """A model missing from the export is a model quietly excluded from a paper."""
    output = render(result)  # type: ignore[operator]
    for name in ("gradient_boosting", "linear", "mlp"):
        # LaTeX escapes the underscore, and must: it is a subscript otherwise.
        assert name in output or name.replace("_", "\\_") in output


@pytest.mark.parametrize("render", RENDERERS)
def test_every_format_reports_the_failure_rather_than_a_score(
    result: BenchResult, render: object
) -> None:
    """A model that did not train has no number, and inventing one would be a lie."""
    output = render(result)  # type: ignore[operator]
    assert "failed" in output


@pytest.mark.parametrize("render", RENDERERS)
def test_every_format_carries_the_tie_into_the_conclusion(
    result: BenchResult, render: object
) -> None:
    """The whole point of the significance test is that it reaches the reader.

    A table showing 0.91 against 0.87 with no note reads as a clear win. It is
    not one: at this sample size the difference is indistinguishable from noise,
    and the export has to say so or it has misrepresented the result.
    """
    output = render(result)  # type: ignore[operator]
    assert "indistinguishable" in output


@pytest.mark.parametrize("render", RENDERERS)
def test_every_format_reports_the_interval_not_just_the_point(
    result: BenchResult, render: object
) -> None:
    """A point estimate without its interval is the number people over-read."""
    output = render(result)  # type: ignore[operator]
    assert "0.88" in output  # the leader's lower bound


def test_a_significant_win_is_stated_as_one() -> None:
    """The other branch of the verdict: no ties, so no hedging."""
    result = BenchResult(
        task="classification",
        target="purchased",
        folds=5,
        primary_metric="balanced_accuracy",
        results=(_model("gradient_boosting", 0.91), _model("linear", 0.62)),
        comparisons=(
            _comparison("linear", "gradient_boosting", significant=True, p=0.004),
        ),
        policy=PreparationPolicy(),
        n_rows=600,
    )
    markdown = render_benchmark_markdown(result)
    assert "statistically significant" in markdown
    assert "indistinguishable" not in markdown


def test_a_benchmark_where_nothing_trained_says_so() -> None:
    """Rendering an empty leaderboard must not raise on `leader is None`."""
    result = BenchResult(
        task="classification",
        target="purchased",
        folds=5,
        primary_metric="balanced_accuracy",
        results=(_model("linear", 0.0, failed="all folds errored"),),
        comparisons=(),
        policy=PreparationPolicy(),
        n_rows=600,
    )
    assert "No model could be trained" in render_benchmark_markdown(result)


def test_a_lower_is_better_metric_is_not_described_as_scoring_highest() -> None:
    """An RMSE leader has the *lowest* number, and the sentence has to match."""
    result = BenchResult(
        task="regression",
        target="price",
        folds=5,
        primary_metric="rmse",
        results=(
            _model("gradient_boosting", 1200.0, metric="rmse"),
            _model("linear", 2400.0, metric="rmse"),
        ),
        comparisons=(
            _comparison("linear", "gradient_boosting", significant=True, p=0.01),
        ),
        policy=PreparationPolicy(),
        n_rows=600,
    )
    markdown = render_benchmark_markdown(result)
    assert "scores lowest" in markdown
    assert "scores highest" not in markdown


# --- format-specific obligations -------------------------------------------


def test_latex_escapes_what_latex_would_otherwise_eat() -> None:
    """An underscore in a model name is a subscript, and the table stops compiling."""
    result = BenchResult(
        task="classification",
        target="signed_up",
        folds=5,
        primary_metric="balanced_accuracy",
        results=(_model("random_forest", 0.9),),
        comparisons=(),
        policy=PreparationPolicy(),
        n_rows=100,
    )
    latex = render_benchmark_latex(result)
    assert "random\\_forest" in latex
    assert "random_forest" not in latex


def test_latex_output_is_a_complete_table_environment() -> None:
    """Half a table is not something anyone can paste into a paper."""
    result = BenchResult(
        task="classification",
        target="purchased",
        folds=5,
        primary_metric="balanced_accuracy",
        results=(_model("linear", 0.9),),
        comparisons=(),
        policy=PreparationPolicy(),
        n_rows=100,
    )
    latex = render_benchmark_latex(result)
    assert latex.count("\\begin{tabular}") == latex.count("\\end{tabular}") == 1


def test_html_export_is_self_contained(result: BenchResult) -> None:
    """A report that fetches a stylesheet is a report that breaks offline."""
    html = render_benchmark_html(result)
    assert html.lstrip().startswith("<!") or "<html" in html
    assert 'src="http' not in html and 'href="http' not in html


# --- the run-comparison exports --------------------------------------------


@pytest.fixture
def groups() -> tuple[object, ...]:
    """Two runs over the same data, which is what makes them comparable."""
    from michi.core.manifest import ModelSpec, RunManifest, SourceInfo
    from michi.report.runs import RunGroup

    def manifest(run_id: str, model: str, value: float) -> RunManifest:
        return RunManifest(
            run_id=run_id,
            kind="eval",
            dataset=SourceInfo("data.csv", "a" * 64, 10, "csv", 600),
            target="purchased",
            task="classification",
            model=ModelSpec(f"{model}.pkl", "joblib", model),
            metrics=(
                Metric(
                    "balanced_accuracy",
                    value,
                    ci_low=value - 0.02,
                    ci_high=value + 0.02,
                ),
            ),
            n_rows=600,
        )

    return (
        RunGroup(
            dataset="data.csv",
            dataset_sha="a" * 64,
            target="purchased",
            task="classification",
            manifests=(
                manifest("20260101T000000Z-aaaa1111", "linear", 0.83),
                manifest("20260102T000000Z-bbbb2222", "forest", 0.89),
            ),
        ),
    )


@pytest.mark.parametrize(
    "render", (render_runs_markdown, render_runs_latex, render_runs_html)
)
def test_run_exports_rank_the_better_run_first(
    groups: tuple[object, ...], render: object
) -> None:
    """`diff` exists to answer which run won; the export must not reorder that."""
    output = render(groups)  # type: ignore[operator]
    assert output.index("forest") < output.index("linear")


@pytest.mark.parametrize(
    "render", (render_runs_markdown, render_runs_latex, render_runs_html)
)
def test_run_exports_identify_the_data_they_compare_over(
    groups: tuple[object, ...], render: object
) -> None:
    """Two runs over different data are not comparable, so the file says which."""
    output = render(groups)  # type: ignore[operator]
    assert "purchased" in output
