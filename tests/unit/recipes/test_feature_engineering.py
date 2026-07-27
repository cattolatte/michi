"""Tests for the feature-engineering recipe operations.

Feature engineering is where a tabular project is usually won, and where
leakage is usually introduced. These tests hold both halves: that each
operation does what it says, and that the ones which learn from data are
classified as fitted, so they land inside the cross-validation fold rather
than in the deterministic pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from michi.core.errors import RecipeError
from michi.recipes import Recipe, RecipeStep, apply_recipe, export_recipe
from michi.recipes.author import command_for, recipe_from_flags


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small frame with a timestamp, a skewed column, and negatives."""
    return pd.DataFrame(
        {
            "signup": [
                "2021-03-04",
                "2022-07-19",
                "2020-01-01",
                "2023-11-30",
                None,
                "2021-06-15",
            ],
            "fare": [1.0, 50.0, 900.0, 12.0, 3.0, 220.0],
            "age": [20.0, 41.0, 33.0, 58.0, 27.0, 64.0],
            "score": [-3.0, 4.0, 0.0, 9.0, float("nan"), 2.0],
            "label": [0, 1, 1, 0, 1, 0],
        }
    )


def _apply(step: RecipeStep, frame: pd.DataFrame) -> pd.DataFrame:
    return apply_recipe(Recipe(steps=(step,)), frame, strict=False).frame


# --- datepart --------------------------------------------------------------


def test_datepart_adds_one_column_per_part(frame: pd.DataFrame) -> None:
    """A timestamp is one enormous integer; the signal is in its parts."""
    result = _apply(
        RecipeStep("datepart", {"columns": ["signup"], "parts": ["year", "month"]}),
        frame,
    )
    assert result["signup_year"].tolist()[:3] == [2021.0, 2022.0, 2020.0]
    assert "signup_month" in result.columns
    assert "signup" in result.columns


def test_datepart_leaves_an_unparseable_timestamp_missing(
    frame: pd.DataFrame,
) -> None:
    """A row michi cannot read a date from gets no invented value."""
    result = _apply(
        RecipeStep("datepart", {"columns": ["signup"], "parts": ["year"]}), frame
    )
    assert pd.isna(result["signup_year"].iloc[4])


def test_datepart_rejects_a_part_it_cannot_extract(frame: pd.DataFrame) -> None:
    """A typo names the parts that do exist rather than silently doing less."""
    with pytest.raises(RecipeError, match="dayofweek"):
        _apply(
            RecipeStep("datepart", {"columns": ["signup"], "parts": ["nonsense"]}),
            frame,
        )


# --- log -------------------------------------------------------------------


def test_log_compresses_a_long_right_tail(frame: pd.DataFrame) -> None:
    """The point of the transform is that the tail stops dominating."""
    before = frame["fare"].skew()
    after = _apply(RecipeStep("log", {"columns": ["fare"]}), frame)["fare"].skew()
    assert after < before


def test_plain_log_cannot_represent_a_negative_value(frame: pd.DataFrame) -> None:
    """log1p of a negative number is undefined; michi must not invent one."""
    result = _apply(RecipeStep("log", {"columns": ["score"]}), frame)
    assert pd.isna(result["score"].iloc[0])


def test_signed_log_keeps_the_sign_of_a_negative_value(
    frame: pd.DataFrame,
) -> None:
    """`signed` is the method for a column that genuinely goes below zero."""
    result = _apply(
        RecipeStep("log", {"columns": ["score"], "method": "signed"}), frame
    )
    assert result["score"].iloc[0] < 0
    assert not pd.isna(result["score"].iloc[0])


# --- interact --------------------------------------------------------------


def test_interact_adds_one_column_per_pair(frame: pd.DataFrame) -> None:
    """Three columns make three pairs, not three columns."""
    result = _apply(
        RecipeStep("interact", {"columns": ["fare", "age", "score"]}), frame
    )
    added = [name for name in result.columns if "_x_" in name]
    assert len(added) == 3


def test_interact_multiplies_the_values(frame: pd.DataFrame) -> None:
    """A linear model cannot represent a product it was not handed."""
    result = _apply(RecipeStep("interact", {"columns": ["fare", "age"]}), frame)
    assert result["fare_x_age"].iloc[0] == pytest.approx(20.0)


def test_a_ratio_by_zero_becomes_missing_not_infinite(
    frame: pd.DataFrame,
) -> None:
    """Most models reject an infinity more confusingly than a missing value."""
    result = _apply(
        RecipeStep("interact", {"columns": ["fare", "score"], "method": "ratio"}),
        frame,
    )
    assert not np.isinf(result["fare_over_score"].astype("float64")).any()


# --- binarize --------------------------------------------------------------


def test_binarize_splits_at_the_threshold(frame: pd.DataFrame) -> None:
    """Above the threshold is 1, at or below it is 0."""
    result = _apply(
        RecipeStep("binarize", {"columns": ["score"], "threshold": 0.0}), frame
    )
    assert result["score"].tolist()[:4] == [0.0, 1.0, 0.0, 1.0]


def test_binarize_leaves_a_missing_value_missing(frame: pd.DataFrame) -> None:
    """A missing value is not "below the threshold" — it is unknown."""
    result = _apply(
        RecipeStep("binarize", {"columns": ["score"], "threshold": 0.0}), frame
    )
    assert pd.isna(result["score"].iloc[4])


# --- bin, and the leakage classification -----------------------------------


def test_bin_produces_at_most_the_requested_number_of_bins(
    frame: pd.DataFrame,
) -> None:
    """Binning discretises; it does not invent levels."""
    result = _apply(RecipeStep("bin", {"columns": ["age"], "bins": 3}), frame)
    assert result["age"].nunique() <= 3


def test_bin_survives_a_column_too_degenerate_to_split() -> None:
    """A constant column cannot form quantile bins; that is not an error."""
    frame = pd.DataFrame({"flat": [1.0] * 10})
    result = _apply(RecipeStep("bin", {"columns": ["flat"], "bins": 4}), frame)
    assert len(result) == 10


def test_bin_is_rejected_when_it_would_produce_one_bin() -> None:
    """Asking for one bin is asking to delete the column."""
    with pytest.raises(RecipeError, match="at least 2"):
        _apply(
            RecipeStep("bin", {"columns": ["a"], "bins": 1}),
            pd.DataFrame({"a": [1.0, 2.0]}),
        )


def test_only_the_learning_steps_are_classified_as_fitted() -> None:
    """This classification is what puts a step inside the CV fold.

    Getting it wrong in the safe direction costs a little accuracy. Getting
    it wrong in the other direction leaks the test fold into training, which
    is silent and produces a model that scores well and generalises badly.
    """
    deterministic = ("datepart", "log", "interact", "binarize")
    assert not any(RecipeStep(op, {"columns": ["a"]}).is_fitted for op in deterministic)
    assert RecipeStep("bin", {"columns": ["a"]}).is_fitted


def test_a_binned_recipe_puts_the_binner_in_the_pipeline_not_prepare() -> None:
    """Quantile edges learned from everything have seen the test fold."""
    recipe = Recipe(steps=(RecipeStep("bin", {"columns": ["age"], "bins": 4}),))
    code = export_recipe(recipe)
    before, _, after = code.partition("def build_pipeline")
    assert "KBinsDiscretizer" in after
    assert "KBinsDiscretizer" not in before.split("def prepare")[-1]


# --- apply and export must agree -------------------------------------------


def _engineered() -> Recipe:
    return Recipe(
        steps=(
            RecipeStep(
                "datepart",
                {"columns": ["signup"], "parts": ["year", "month", "week"]},
            ),
            RecipeStep("log", {"columns": ["fare"], "method": "signed"}),
            RecipeStep("binarize", {"columns": ["score"], "threshold": 0.5}),
            RecipeStep("interact", {"columns": ["fare", "age"]}),
        ),
        target="label",
    )


def test_apply_and_export_agree_on_every_engineered_op(
    frame: pd.DataFrame, tmp_path: Path
) -> None:
    """A user who leaves michi must get exactly the transform they had."""
    import importlib.util

    recipe = _engineered()
    module_path = tmp_path / "engineered.py"
    module_path.write_text(export_recipe(recipe), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("engineered", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pd.testing.assert_frame_equal(
        module.prepare(frame).reset_index(drop=True),
        apply_recipe(recipe, frame, strict=False).frame.reset_index(drop=True),
        check_dtype=False,
    )


def test_engineered_code_passes_ruff(tmp_path: Path) -> None:
    """Generated feature engineering satisfies the same linter michi does."""
    path = tmp_path / "pipeline.py"
    recipe = Recipe(
        steps=(
            *_engineered().steps,
            RecipeStep("bin", {"columns": ["age"], "bins": 4}),
        ),
        target="label",
    )
    path.write_text(export_recipe(recipe), encoding="utf-8")
    for command in (["check"], ["format", "--check"]):
        result = subprocess.run(
            [sys.executable, "-m", "ruff", *command, str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 1 and "No module named" in result.stderr:
            pytest.skip("ruff is not installed in this environment")
        assert result.returncode == 0, result.stdout


# --- flag parity -----------------------------------------------------------


def test_every_engineered_op_round_trips_through_its_command() -> None:
    """ADR: the printed command must rebuild the recipe it was printed from.

    A step the wizard can produce but the flags cannot is a capability that
    exists only interactively, which is invisible to scripts and to CI.
    """
    original = recipe_from_flags(
        None,
        datepart=[("signup", "year+month")],
        log=[("fare", "log1p")],
        interact=["fare", "age"],
        binarize=[("score", "0.0")],
        bin_=[("age", "4:quantile")],
        target="label",
    )
    command = command_for(original, "d.csv")
    for fragment in (
        "--datepart signup=year+month",
        "--log fare=log1p",
        "--interact fare,age",
        "--binarize score=0.0",
        "--bin age=4:quantile",
    ):
        assert fragment in command


def test_derived_columns_come_after_the_values_they_derive_from() -> None:
    """Ordering is mechanical: cast a date before reading parts off it."""
    recipe = recipe_from_flags(
        None,
        cast=[("signup", "datetime")],
        datepart=[("signup", "year")],
        interact=["fare", "age"],
        log=[("fare", "log1p")],
    )
    order = [step.op for step in recipe.steps]
    assert order.index("cast") < order.index("datepart")
    assert order.index("log") < order.index("interact")


# --- the frozen schema -----------------------------------------------------


def test_a_recipe_written_before_these_ops_still_loads() -> None:
    """Adding vocabulary is additive; existing recipes keep their meaning."""
    payload = {
        "schema_version": "1.0",
        "steps": [{"op": "drop", "columns": ["id"]}],
    }
    assert Recipe.from_dict(payload).steps[0].op == "drop"


def test_an_unknown_op_names_the_ones_that_exist() -> None:
    """An older michi meeting a newer recipe fails loudly, not silently."""
    with pytest.raises(RecipeError, match="datepart"):
        RecipeStep("polynomial", {"columns": ["city"]})


# --- target encoding, and the leak it exists to avoid ----------------------


def _noise_frame(rows: int = 600) -> pd.DataFrame:
    """A high-cardinality id with no relationship to the label at all."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user": [f"u{i}" for i in range(rows)],
            "label": (rng.random(rows) < 0.5).astype(int),
        }
    )


def test_target_encoding_is_reproducible() -> None:
    """Two identical runs must give two identical numbers.

    scikit-learn's own TargetEncoder is not, on any version michi supports:
    the parameter that made it deterministic is deprecated in 1.9 and removed
    in 1.11. That is why michi owns this one.
    """
    from michi.recipes.encoders import encode_frame

    frame = _noise_frame(200)
    first = encode_frame(frame, ["user"], frame["label"])
    second = encode_frame(frame, ["user"], frame["label"])
    assert np.allclose(first, second)


def test_target_encoding_does_not_memorise_the_label() -> None:
    """The whole point: a pure-noise id must stay pure noise.

    Encoded naively, a unique id maps one-to-one onto its own label and a
    model scores a perfect 1.0 on data containing no signal whatsoever. Out
    of fold, the same id predicts nothing, which is the truth.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    from michi.recipes.encoders import build_target_encoder

    frame = _noise_frame()
    features, labels = frame[["user"]], frame["label"]

    honest = cross_val_score(
        Pipeline([("encode", build_target_encoder()), ("model", LogisticRegression())]),
        features,
        labels,
        cv=5,
    ).mean()

    leaked_column = frame["user"].map(labels.groupby(frame["user"]).mean())
    leaked = cross_val_score(
        LogisticRegression(), leaked_column.to_frame(), labels, cv=5
    ).mean()

    assert leaked > 0.95, "the naive encoding should look perfect — that is the trap"
    assert honest < 0.65, f"out-of-fold encoding leaked: scored {honest}"


def test_an_unseen_category_falls_back_to_the_prior() -> None:
    """A category the training folds never saw carries no information."""
    from michi.recipes.encoders import build_target_encoder

    encoder = build_target_encoder()
    train = pd.DataFrame({"c": ["a", "a", "b", "b", "a", "b"]})
    encoder.fit(train, pd.Series([1, 1, 0, 0, 1, 0]))
    encoded = encoder.transform(pd.DataFrame({"c": ["never-seen"]}))
    assert encoded[0][0] == pytest.approx(encoder.prior_)


def test_target_encoding_needs_its_target_named() -> None:
    """A recipe that cannot be reproduced from the file alone is not a recipe."""
    with pytest.raises(RecipeError, match="target"):
        _apply(RecipeStep("target-encode", {"columns": ["user"]}), _noise_frame(50))


def test_exported_code_carries_the_encoder_rather_than_importing_michi() -> None:
    """The generated file promises pandas and scikit-learn only."""
    recipe = Recipe(
        steps=(RecipeStep("target-encode", {"columns": ["user"], "target": "label"}),),
        target="label",
    )
    code = export_recipe(recipe)
    assert "class OutOfFoldTargetEncoder" in code
    # Prose may mention michi; code may not import it. That distinction is
    # the promise: the file runs after michi is uninstalled.
    imports = [
        line
        for line in code.splitlines()
        if line.startswith(("import ", "from ")) and "michi" in line
    ]
    assert imports == []


# --- the feature menu ------------------------------------------------------


def test_opportunities_are_offered_only_where_the_shape_fits(
    messy_csv: Path,
) -> None:
    """Contextual menus: no timestamp, no datepart offer."""
    from michi.core.io import load_table
    from michi.inspection import profile_table
    from michi.recipes import feature_opportunities

    profile = profile_table(load_table(messy_csv), target="purchased")
    offered = {item.op for item in feature_opportunities(profile)}
    kinds = {column.kind.value for column in profile.columns}
    if "datetime" not in kinds:
        assert "datepart" not in offered
    assert offered, "a dataset with numeric columns should offer something"


def test_an_opportunity_builds_a_step_for_only_the_chosen_columns(
    messy_csv: Path,
) -> None:
    """Selecting two of five columns must not quietly engineer all five."""
    from michi.core.io import load_table
    from michi.inspection import profile_table
    from michi.recipes import feature_opportunities

    profile = profile_table(load_table(messy_csv), target="purchased")
    opportunity = next(
        item for item in feature_opportunities(profile) if len(item.columns) >= 2
    )
    step = opportunity.step(list(opportunity.columns[:1]))
    assert step.columns == opportunity.columns[:1]
    assert step.why


def test_the_menu_never_preselects(messy_csv: Path) -> None:
    """michi lists shapes and stays quiet about worth.

    An opportunity says "these columns are the right shape for this", never
    "you should do this" — so nothing in it may carry a default that reads as
    a recommendation.
    """
    from michi.core.io import load_table
    from michi.inspection import profile_table
    from michi.recipes import feature_opportunities

    profile = profile_table(load_table(messy_csv), target="purchased")
    for item in feature_opportunities(profile):
        assert not hasattr(item, "default")
        assert "should" not in item.detail.lower()
        assert "recommend" not in item.detail.lower()


def test_target_encoding_the_target_itself_is_refused() -> None:
    """A step that encodes the label with its own mean is a no-op in disguise.

    Dropping it silently would leave a recipe claiming to do something it does
    not, which is worse than the mistake it hides.
    """
    with pytest.raises(RecipeError, match="cannot encode the target itself"):
        recipe_from_flags(None, target_encode=["purchased"], target="purchased")


def test_target_encoding_without_a_target_is_refused() -> None:
    """The encoding is against a label; there is nothing to encode without one."""
    with pytest.raises(RecipeError, match="needs --target"):
        recipe_from_flags(None, target_encode=["city"])


def test_target_encoding_round_trips_through_its_command() -> None:
    """Flag parity holds for the newest operation too."""
    recipe = recipe_from_flags(None, target_encode=["city", "postcode"], target="y")
    assert "--target-encode city,postcode" in command_for(recipe, "d.csv")
