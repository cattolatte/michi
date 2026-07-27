"""Tests for `diff`, `threshold`, and `errors`.

Each exists because something is decided silently: whether the data still
resembles the baseline, where the decision cutoff sits, and which rows the
score is made of.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from typer.testing import CliRunner

from michi.cli.app import app
from michi.core.artifacts import Severity
from michi.core.io import load_table
from michi.inspection import profile_table
from michi.inspection.drift import compare_profiles

runner = CliRunner()


@pytest.fixture
def baseline(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = 300
    signal = rng.normal(size=rows)
    pd.DataFrame(
        {
            "signal": signal,
            "region": rng.choice(["north", "south"], size=rows),
            "label": (signal + rng.normal(scale=0.4, size=rows) > 0).astype(int),
        }
    ).to_csv(tmp_path / "base.csv", index=False)
    return tmp_path / "base.csv"


def _profile(path: Path) -> object:
    return profile_table(load_table(path), target="label")


# --- diff ------------------------------------------------------------------


def test_a_removed_column_is_the_most_severe_change(
    baseline: Path, tmp_path: Path
) -> None:
    """It breaks the model today, whatever the distributions did."""
    frame = pd.read_csv(baseline).drop(columns=["region"])
    frame.to_csv(tmp_path / "now.csv", index=False)
    report = compare_profiles(_profile(baseline), _profile(tmp_path / "now.csv"))
    assert report.worst is Severity.HIGH
    assert any(item.kind == "column-removed" for item in report.findings)


def test_a_new_column_is_information_not_a_warning(
    baseline: Path, tmp_path: Path
) -> None:
    """A fitted model never asked for it, so it cannot break anything."""
    frame = pd.read_csv(baseline).assign(extra=1.0)
    frame.to_csv(tmp_path / "now.csv", index=False)
    report = compare_profiles(_profile(baseline), _profile(tmp_path / "now.csv"))
    added = next(item for item in report.findings if item.kind == "column-added")
    assert added.severity is Severity.INFO


def test_a_shifted_mean_is_measured_in_baseline_deviations(
    baseline: Path, tmp_path: Path
) -> None:
    """A threshold in raw units would mean different things per column."""
    frame = pd.read_csv(baseline)
    frame["signal"] = frame["signal"] + 3.0
    frame.to_csv(tmp_path / "now.csv", index=False)
    report = compare_profiles(_profile(baseline), _profile(tmp_path / "now.csv"))
    shift = next(
        item for item in report.findings if item.kind == "distribution-shifted"
    )
    assert shift.metrics["shift_sd"] > 2


def test_an_unchanged_dataset_reports_nothing(baseline: Path) -> None:
    """Silence is the correct output when nothing moved."""
    report = compare_profiles(_profile(baseline), _profile(baseline))
    assert report.findings == ()


def test_a_new_category_is_flagged(baseline: Path, tmp_path: Path) -> None:
    """An encoder has no slot for a level it never saw."""
    frame = pd.read_csv(baseline)
    frame.loc[frame.index[:60], "region"] = "WEST_NEW"
    frame.to_csv(tmp_path / "now.csv", index=False)
    report = compare_profiles(_profile(baseline), _profile(tmp_path / "now.csv"))
    assert any(item.kind == "new-categories" for item in report.findings)


def test_diff_gates_on_severity(baseline: Path, tmp_path: Path) -> None:
    """The command a person runs by hand is the one CI runs nightly."""
    pd.read_csv(baseline).drop(columns=["region"]).to_csv(
        tmp_path / "now.csv", index=False
    )
    result = runner.invoke(
        app, ["diff", str(baseline), str(tmp_path / "now.csv"), "--fail-on", "high"]
    )
    assert result.exit_code == 1


def test_diff_accepts_a_committed_profile_as_the_baseline(
    baseline: Path, tmp_path: Path
) -> None:
    """A profile.json is small, diffable, and already in the repository."""
    profile_json = tmp_path / "p.json"
    runner.invoke(
        app,
        ["inspect", str(baseline), "--target", "label", "--json", str(profile_json)],
    )
    result = runner.invoke(app, ["diff", str(profile_json), str(baseline)])
    assert result.exit_code == 0, result.output
    assert "Nothing moved" in result.output


def test_a_json_file_that_is_not_a_profile_says_so(tmp_path: Path) -> None:
    """An error names the command that would produce the right file."""
    stray = tmp_path / "stray.json"
    stray.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    result = runner.invoke(app, ["diff", str(stray), str(stray)])
    assert result.exit_code == 2
    assert "michi inspect" in " ".join(result.output.split())


# --- threshold -------------------------------------------------------------


@pytest.fixture
def fitted(baseline: Path, tmp_path: Path) -> Path:
    frame = pd.read_csv(baseline)
    model = LogisticRegression().fit(frame[["signal"]], frame["label"])
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    frame[["signal", "label"]].to_csv(tmp_path / "scored.csv", index=False)
    return path


def test_the_whole_curve_is_shown(fitted: Path, tmp_path: Path) -> None:
    """michi prints what every cutoff buys; the user picks."""
    result = runner.invoke(
        app,
        [
            "threshold",
            str(fitted),
            str(tmp_path / "scored.csv"),
            "--target",
            "label",
            "--steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    combined = " ".join(result.output.split())
    assert "precision" in combined and "recall" in combined


def test_costs_move_the_marked_cutoff(fitted: Path, tmp_path: Path) -> None:
    """A miss costing ten times a false alarm should lower the cutoff.

    This is the whole point: 0.5 is right only when the two errors cost the
    same, and nothing else says so.
    """
    cheap = runner.invoke(
        app,
        [
            "threshold",
            str(fitted),
            str(tmp_path / "scored.csv"),
            "--target",
            "label",
            "--cost",
            "fn=1,fp=1",
            "--objective",
            "cost",
            "--steps",
            "9",
        ],
    )
    dear = runner.invoke(
        app,
        [
            "threshold",
            str(fitted),
            str(tmp_path / "scored.csv"),
            "--target",
            "label",
            "--cost",
            "fn=20,fp=1",
            "--objective",
            "cost",
            "--steps",
            "9",
        ],
    )
    assert cheap.exit_code == 0 and dear.exit_code == 0
    assert _marked(cheap.output) > _marked(dear.output)


def _marked(output: str) -> float:
    """The cutoff michi marked, read back out of the table."""
    for line in output.splitlines():
        if "▸" in line:
            return float(line.split()[1])
    raise AssertionError("no cutoff was marked")


def test_a_cost_objective_without_costs_is_refused(
    fitted: Path, tmp_path: Path
) -> None:
    """Minimising an unstated cost would be michi inventing the weights."""
    result = runner.invoke(
        app,
        [
            "threshold",
            str(fitted),
            str(tmp_path / "scored.csv"),
            "--target",
            "label",
            "--objective",
            "cost",
        ],
    )
    assert result.exit_code == 2
    assert "--cost" in result.output


def test_a_malformed_cost_names_the_shape(fitted: Path, tmp_path: Path) -> None:
    """An error says what to type, not only that something was wrong."""
    result = runner.invoke(
        app,
        [
            "threshold",
            str(fitted),
            str(tmp_path / "scored.csv"),
            "--target",
            "label",
            "--cost",
            "nonsense",
        ],
    )
    assert result.exit_code == 2
    assert "fn=" in result.output


# --- errors ----------------------------------------------------------------


def test_mistakes_are_ordered_by_confidence(fitted: Path, tmp_path: Path) -> None:
    """A confident error is a leak or a blind spot; an unsure one is the edge."""
    result = runner.invoke(
        app,
        [
            "errors",
            str(fitted),
            str(tmp_path / "scored.csv"),
            "--target",
            "label",
            "--show",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ordered by how sure" in " ".join(result.output.split())


def test_the_mistakes_can_be_written_out(fitted: Path, tmp_path: Path) -> None:
    """The useful next step is opening them somewhere that is not a terminal."""
    out = tmp_path / "wrong.csv"
    result = runner.invoke(
        app,
        [
            "errors",
            str(fitted),
            str(tmp_path / "scored.csv"),
            "--target",
            "label",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "predicted" in pd.read_csv(out).columns


def test_a_perfect_model_reports_no_mistakes(tmp_path: Path) -> None:
    """No errors is a result, not an empty table."""
    frame = pd.DataFrame({"x": [0.0, 1.0] * 30, "label": [0, 1] * 30})
    frame.to_csv(tmp_path / "easy.csv", index=False)
    model = LogisticRegression().fit(frame[["x"]], frame["label"])
    path = tmp_path / "perfect.joblib"
    joblib.dump(model, path)
    result = runner.invoke(
        app, ["errors", str(path), str(tmp_path / "easy.csv"), "--target", "label"]
    )
    assert "No mistakes" in result.output
