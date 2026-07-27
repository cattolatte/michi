"""Tests for the last mile: `tune`, `fit`, and `predict`.

These three close the loop michi used to stop one step short of. Everything
before them compares and explains; these produce a model and a file of
predictions — which is what a competition submission, a batch scoring job,
and a smoke test all need.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from michi.cli.app import app

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small, learnable classification dataset written to disk."""
    import numpy as np

    rng = np.random.default_rng(0)
    rows = 240
    signal = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "id": range(rows),
            "signal": signal,
            "noise": rng.normal(size=rows),
            "group": rng.choice(["a", "b", "c"], size=rows),
            "label": (signal + rng.normal(scale=0.3, size=rows) > 0).astype(int),
        }
    )
    frame.to_csv(tmp_path / "train.csv", index=False)
    frame.drop(columns=["label"]).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


# --- fit -------------------------------------------------------------------


def test_fit_writes_a_loadable_model(project: Path) -> None:
    """The artifact `fit` produces must be one `predict` can consume."""
    import joblib

    destination = project / "model.joblib"
    result = runner.invoke(
        app,
        [
            "fit",
            str(project / "train.csv"),
            "--target",
            "label",
            "--model",
            "tree",
            "-o",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    assert destination.exists()
    assert hasattr(joblib.load(destination), "predict")


def test_fit_reports_no_score(project: Path) -> None:
    """A score on the rows a model trained on is confidently wrong.

    Printing one would be the most misleading number in the toolbox, so `fit`
    prints none and says where to get an honest one instead.
    """
    result = runner.invoke(
        app,
        [
            "fit",
            str(project / "train.csv"),
            "--target",
            "label",
            "-o",
            str(project / "m.joblib"),
        ],
    )
    combined = " ".join(result.output.split())
    assert "michi bench" in combined
    assert "accuracy" not in combined.lower()


def test_fit_without_a_target_is_actionable(project: Path) -> None:
    """Missing a target names both ways to supply one."""
    result = runner.invoke(app, ["fit", str(project / "train.csv")])
    assert result.exit_code == 2
    assert "michi.toml" in " ".join(result.output.split())


# --- predict ---------------------------------------------------------------


def _fit(project: Path, name: str = "model.joblib") -> Path:
    destination = project / name
    runner.invoke(
        app,
        [
            "fit",
            str(project / "train.csv"),
            "--target",
            "label",
            "--model",
            "tree",
            "-o",
            str(destination),
        ],
    )
    return destination


def test_predict_needs_no_labels(project: Path) -> None:
    """The whole point: a held-out file has no answer column."""
    model = _fit(project)
    output = project / "preds.csv"
    result = runner.invoke(
        app, ["predict", str(model), str(project / "test.csv"), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    predictions = pd.read_csv(output)
    assert len(predictions) == 240
    assert "prediction" in predictions.columns


def test_predict_carries_an_id_column_through(project: Path) -> None:
    """A submission file is useless without the key it joins on."""
    model = _fit(project)
    output = project / "submission.csv"
    runner.invoke(
        app,
        [
            "predict",
            str(model),
            str(project / "test.csv"),
            "--id",
            "id",
            "-o",
            str(output),
        ],
    )
    submission = pd.read_csv(output)
    assert list(submission.columns)[:2] == ["id", "prediction"]
    assert submission["id"].tolist() == list(range(240))


def test_predict_writes_probabilities_when_asked(project: Path) -> None:
    """Ranked submissions need scores, not labels."""
    model = _fit(project)
    output = project / "proba.csv"
    runner.invoke(
        app,
        [
            "predict",
            str(model),
            str(project / "test.csv"),
            "--proba",
            "-o",
            str(output),
        ],
    )
    columns = pd.read_csv(output).columns
    assert any(name.startswith("proba_") for name in columns)


def test_predict_tolerates_a_label_column_that_is_still_present(
    project: Path,
) -> None:
    """Held-out files often keep an empty label column; that must not fail."""
    model = _fit(project)
    labelled = pd.read_csv(project / "train.csv")
    labelled.to_csv(project / "with_label.csv", index=False)
    result = runner.invoke(
        app,
        [
            "predict",
            str(model),
            str(project / "with_label.csv"),
            "--drop-target",
            "label",
            "-o",
            str(project / "p.csv"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_predict_accepts_any_model_with_a_predict_method(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deep learning arrives through the protocol, not a per-framework loader.

    A torch or TensorFlow model is reached exactly this way: michi calls
    `predict` and never inspects what is behind it.
    """
    module = project / "netstub.py"
    module.write_text(
        "import numpy as np\n"
        "class Net:\n"
        "    classes_ = np.array([0, 1])\n"
        "    def predict(self, frame):\n"
        "        return np.zeros(len(frame), dtype=int)\n"
        "model = Net()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(project))
    output = project / "dl.csv"
    result = runner.invoke(
        app, ["predict", "netstub:model", str(project / "test.csv"), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output)) == 240


# --- tune ------------------------------------------------------------------


def test_the_search_space_is_printable_before_anything_runs() -> None:
    """A space you cannot inspect is a decision michi made out of sight."""
    result = runner.invoke(app, ["tune", "--model", "rf", "--list-space"])
    assert result.exit_code == 0
    combined = " ".join(result.output.split())
    assert "max_depth" in combined
    assert "combinations" in combined


def test_an_unknown_model_names_the_ones_with_spaces() -> None:
    """An error says what to do, not only what went wrong."""
    result = runner.invoke(app, ["tune", "--model", "nonesuch", "--list-space"])
    assert result.exit_code == 2
    assert "ridge" in " ".join(result.output.split())


def test_tuning_reports_the_held_out_score_and_the_optimism(
    project: Path,
) -> None:
    """The gap between the search's own best and the honest score is the lesson.

    Reporting the inner score as performance is the second most common silent
    leak in tabular ML; michi prints both, side by side.
    """
    result = runner.invoke(
        app,
        [
            "tune",
            str(project / "train.csv"),
            "--target",
            "label",
            "--model",
            "tree",
            "--candidates",
            "4",
            "--cv",
            "2",
            "--inner-cv",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    combined = " ".join(result.output.split())
    assert "tuned (held-out folds)" in combined
    assert "search's own best" in combined
    assert "defaults (same folds)" in combined


def test_tuned_parameters_round_trip_into_fit(project: Path) -> None:
    """`tune --save-params` must produce something `fit --params` accepts.

    Two verbs that cannot hand off to each other are two verbs that make the
    user do the joining by hand.
    """
    params = project / "best.yaml"
    runner.invoke(
        app,
        [
            "tune",
            str(project / "train.csv"),
            "--target",
            "label",
            "--model",
            "tree",
            "--candidates",
            "4",
            "--cv",
            "2",
            "--inner-cv",
            "2",
            "--save-params",
            str(params),
        ],
    )
    assert params.exists()

    result = runner.invoke(
        app,
        [
            "fit",
            str(project / "train.csv"),
            "--target",
            "label",
            "--model",
            "tree",
            "--params",
            str(params),
            "-o",
            str(project / "tuned.joblib"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_a_user_supplied_space_replaces_the_built_in_one(project: Path) -> None:
    """Bare parameter names work: knowing michi's step prefix is not required."""
    space = project / "space.yaml"
    space.write_text("max_depth: [2, 3]\n", encoding="utf-8")
    result = runner.invoke(
        app, ["tune", "--model", "tree", "--space", str(space), "--list-space"]
    )
    assert result.exit_code == 0
    combined = " ".join(result.output.split())
    assert "max_depth" in combined
    assert "min_samples_leaf" not in combined


# --- bayesian search, and the constraints ADR-0004 put on it ---------------


def test_bayes_is_offered_as_a_strategy() -> None:
    """ADR-0004: a model-based optimiser chooses the order, not the space."""
    from michi.bench.tuning import STRATEGIES

    assert "bayes" in STRATEGIES


def test_an_unavailable_optimiser_never_falls_back_silently() -> None:
    """A user who asked for `bayes` and got `random` compared two experiments.

    ADR-0004 constraint 5: degrade to a *named* alternative. Silently
    substituting a different sampler produces two runs that were never the
    same thing, reported as if they were.
    """
    import numpy as np
    import pytest as _pytest

    from michi.bench.tuning import _build_search
    from michi.core.errors import RunError

    try:
        import optuna  # noqa: F401
    except ImportError:
        with _pytest.raises(RunError, match=r"komichi\[bayes\]"):
            _build_search(
                None,
                {"model__max_depth": [1, 2]},
                strategy="bayes",
                candidates=2,
                folds=2,
                seed=0,
                task="classification",
                labels=np.array([0, 1, 0, 1]),
                metric="balanced_accuracy",
                greater_is_better=True,
            )


def test_every_strategy_reports_the_optimism_gap(project: Path) -> None:
    """ADR-0004 constraint 3: the gap grows with the optimiser's strength.

    Reporting an inner search score as performance gets *more* wrong the
    better the search gets, which is the worst direction for an error to
    move — so the honest number and the flattering one are always adjacent.
    """
    result = runner.invoke(
        app,
        [
            "tune",
            str(project / "train.csv"),
            "--target",
            "label",
            "--model",
            "tree",
            "--candidates",
            "4",
            "--cv",
            "2",
            "--inner-cv",
            "2",
        ],
    )
    combined = " ".join(result.output.split())
    assert "search's own best" in combined
    assert "would have hidden" in combined


def test_the_search_space_is_unchanged_by_the_strategy() -> None:
    """ADR-0004 constraint 1: an optimiser may not widen the user's space."""
    from michi.bench.tuning import search_space

    space = search_space("tree")
    assert set(space) == {"model__max_depth", "model__min_samples_leaf"}


def test_a_count_of_evaluations_survives_either_search_shape() -> None:
    """sklearn records `cv_results_["params"]`; Optuna records `trials_`.

    Reading only the sklearn shape raised KeyError the first time a Bayesian
    search finished a fold.
    """
    from michi.bench.tuning import _evaluated

    class _Sklearn:
        def __init__(self) -> None:
            self.cv_results_ = {"params": [{}, {}, {}]}

    class _Optuna:
        def __init__(self) -> None:
            self.trials_ = [object(), object()]
            self.cv_results_ = {"mean_test_score": [0.1, 0.2]}

    class _Neither:
        def __init__(self) -> None:
            self.cv_results_: dict[str, object] = {}

    assert _evaluated(_Sklearn()) == 3
    assert _evaluated(_Optuna()) == 2
    assert _evaluated(_Neither()) == 0
