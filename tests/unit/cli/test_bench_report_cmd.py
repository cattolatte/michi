"""Tests for the `michi bench` and `michi report` commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from michi.cli.app import app

runner = CliRunner()

# --- the model menu --------------------------------------------------------


def test_list_models_prints_the_menu() -> None:
    """`--list-models` shows what is available without needing a dataset."""
    result = runner.invoke(app, ["bench", "--list-models"])
    assert result.exit_code == 0
    assert "rf" in result.output
    assert "dummy" in result.output


def test_list_models_can_be_filtered_by_task() -> None:
    """The menu narrows to models that support the task."""
    result = runner.invoke(app, ["bench", "--list-models", "--task", "regression"])
    assert "lasso" in result.output


def test_bench_without_a_dataset_explains_itself() -> None:
    """Missing arguments produce a usage error, not a traceback."""
    result = runner.invoke(app, ["bench"])
    assert result.exit_code != 0


# --- running a benchmark ---------------------------------------------------


def test_bench_reports_a_leaderboard(tidy_csv: Path) -> None:
    """The default run ranks the models and names the metric."""
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear,tree",
            "--cv",
            "3",
            "--no-save",
        ],
    )
    assert result.exit_code == 0
    assert "balanced_accuracy" in result.output
    assert "dummy" in result.output


def test_bench_states_its_preparation(tidy_csv: Path) -> None:
    """What michi does to the columns is printed, never hidden."""
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "tree",
            "--cv",
            "3",
            "--no-save",
        ],
    )
    assert "preparation:" in result.output
    assert "fitted inside each fold" in result.output


def test_bench_prints_a_plain_language_verdict(tidy_csv: Path) -> None:
    """The conclusion is a sentence, not a p-value the reader must decode."""
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear,tree",
            "--cv",
            "3",
            "--no-save",
        ],
    )
    assert "Verdict" in result.output


def test_bench_writes_one_manifest_per_model(tidy_csv: Path, tmp_path: Path) -> None:
    """Each model's result is separately recorded."""
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear,tree",
            "--cv",
            "3",
            "--runs-dir",
            str(runs),
        ],
    )
    assert result.exit_code == 0
    assert len(list(runs.glob("*.json"))) == 3  # linear, tree, dummy


def test_bench_writes_an_html_report(tidy_csv: Path, tmp_path: Path) -> None:
    """`--report` produces a self-contained offline page."""
    destination = tmp_path / "bench.html"
    runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "tree",
            "--cv",
            "3",
            "--no-save",
            "--report",
            str(destination),
        ],
    )
    html = destination.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "http://" not in html
    assert "https://" not in html


def test_unknown_model_exits_with_error(tidy_csv: Path) -> None:
    """A model typo fails immediately, before any training."""
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "randomforest",
            "--cv",
            "3",
            "--no-save",
        ],
    )
    assert result.exit_code == 2


# --- reporting -------------------------------------------------------------


def _make_runs(tidy_csv: Path, runs: Path) -> None:
    runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear,tree",
            "--cv",
            "3",
            "--runs-dir",
            str(runs),
        ],
    )


def test_report_renders_recorded_runs(tidy_csv: Path, tmp_path: Path) -> None:
    """`michi report` summarises whatever is in the runs directory."""
    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    result = runner.invoke(app, ["report", str(runs)])
    assert result.exit_code == 0
    assert "michi report" in result.output


def test_report_writes_html(tidy_csv: Path, tmp_path: Path) -> None:
    """HTML output is a single offline file."""
    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    destination = tmp_path / "report.html"
    result = runner.invoke(app, ["report", str(runs), "--out", str(destination)])
    assert result.exit_code == 0
    html = destination.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "https://" not in html


def test_report_writes_markdown(tidy_csv: Path, tmp_path: Path) -> None:
    """Markdown output suits a pull request or a README."""
    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    result = runner.invoke(app, ["report", str(runs), "--format", "markdown"])
    assert result.exit_code == 0
    assert "| Run |" in result.output


def test_report_writes_latex(tidy_csv: Path, tmp_path: Path) -> None:
    """LaTeX output is a booktabs table ready for a paper."""
    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    result = runner.invoke(app, ["report", str(runs), "--format", "latex"])
    assert result.exit_code == 0
    assert "\\begin{tabular}" in result.output
    assert "\\toprule" in result.output


def test_report_on_an_empty_directory_is_actionable(tmp_path: Path) -> None:
    """With nothing recorded, michi says what to run first."""
    empty = tmp_path / "runs"
    empty.mkdir()
    result = runner.invoke(app, ["report", str(empty)])
    assert result.exit_code == 2
    assert "michi eval" in result.output or "michi bench" in result.output


def test_report_skips_unreadable_manifests(tidy_csv: Path, tmp_path: Path) -> None:
    """One corrupt file must not stop a report over the good ones."""
    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    (runs / "broken.json").write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["report", str(runs)])
    assert result.exit_code == 0


def test_report_groups_by_dataset_and_target(
    tidy_csv: Path, messy_csv: Path, tmp_path: Path
) -> None:
    """Runs over different data are never mixed into one comparison."""
    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    runner.invoke(
        app,
        [
            "bench",
            str(messy_csv),
            "--target",
            "purchased",
            "--models",
            "tree",
            "--cv",
            "3",
            "--runs-dir",
            str(runs),
        ],
    )
    result = runner.invoke(app, ["report", str(runs), "--format", "markdown"])
    assert result.output.count("## ") == 2


def test_unknown_format_is_rejected(tidy_csv: Path, tmp_path: Path) -> None:
    """An unsupported format names the ones that exist."""
    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    result = runner.invoke(app, ["report", str(runs), "--format", "pdf"])
    assert result.exit_code != 0


def test_manifests_round_trip_through_the_report(
    tidy_csv: Path, tmp_path: Path
) -> None:
    """Every written manifest is readable back into an artifact."""
    from michi.core.manifest import RunManifest

    runs = tmp_path / "runs"
    _make_runs(tidy_csv, runs)
    for path in runs.glob("*.json"):
        manifest = RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert manifest.kind == "bench"


# --- the report a benchmark writes -----------------------------------------


def test_report_suffix_picks_the_format(tidy_csv: Path, tmp_path: Path) -> None:
    """One flag, three formats, chosen by the path the user already typed."""
    expected = {
        "out.html": "<",
        "out.md": "# Benchmark",
        "out.tex": "\\begin{tabular}",
    }
    for name, marker in expected.items():
        destination = tmp_path / name
        result = runner.invoke(
            app,
            [
                "bench",
                str(tidy_csv),
                "--target",
                "label",
                "--models",
                "linear",
                "--no-save",
                "--report",
                str(destination),
            ],
        )
        assert result.exit_code == 0, result.output
        assert marker in destination.read_text(encoding="utf-8")


def test_an_unreadable_suffix_fails_before_anything_trains(
    tidy_csv: Path, tmp_path: Path
) -> None:
    """Learning this after a long benchmark, with nothing written, is unusable."""
    destination = tmp_path / "out.xyz"
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear",
            "--no-save",
            "--report",
            str(destination),
        ],
    )
    assert result.exit_code != 0
    assert ".tex" in result.output  # the message names what would have worked
    assert not destination.exists()
    assert "leader" not in result.output  # no benchmark was run


def test_open_on_a_non_html_report_says_so(tidy_csv: Path, tmp_path: Path) -> None:
    """Silently ignoring a flag leaves the user waiting for a browser."""
    destination = tmp_path / "out.tex"
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear",
            "--no-save",
            "--report",
            str(destination),
            "--open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "nothing was opened" in result.output
