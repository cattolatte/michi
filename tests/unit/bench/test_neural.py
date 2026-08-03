"""The neural catalogue entries, including the PyTorch training loop.

`torch` is not a dev dependency — installing it on three operating systems and
two Python versions to test one module would dominate CI. These tests skip
without it and run in the dedicated `torch` job, which is the only place the
loop's body executes at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from michi.bench.neural import build_mlp, build_torch_mlp
from michi.core.errors import RunError


@pytest.fixture
def classification() -> tuple[np.ndarray, np.ndarray]:
    """A separable two-class problem large enough for a validation split."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(160, 6)).astype("float32")
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    return features, labels


@pytest.fixture
def regression() -> tuple[np.ndarray, np.ndarray]:
    """A linear target, so a network that learns anything at all beats the mean."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(160, 6)).astype("float32")
    values = features[:, 0] * 3.0 - features[:, 1]
    return features, values


# --- the scikit-learn rung --------------------------------------------------


def test_mlp_needs_no_extra_dependency(
    classification: tuple[np.ndarray, np.ndarray],
) -> None:
    """The first network someone tries should not require an install."""
    features, labels = classification
    model = build_mlp("classification", seed=0)
    model.fit(features, labels)

    assert model.score(features, labels) > 0.6


def test_mlp_regression_is_a_regressor_not_a_classifier(
    regression: tuple[np.ndarray, np.ndarray],
) -> None:
    """The task string picks the estimator; getting it wrong fails much later."""
    features, values = regression
    model = build_mlp("regression", seed=0)
    model.fit(features, values)

    assert model.predict(features[:4]).dtype.kind == "f"


def test_mlp_is_reproducible_from_its_seed(
    classification: tuple[np.ndarray, np.ndarray],
) -> None:
    """A benchmark that cannot be repeated is not a benchmark."""
    features, labels = classification
    first = build_mlp("classification", seed=7).fit(features, labels)
    second = build_mlp("classification", seed=7).fit(features, labels)

    assert np.array_equal(first.predict(features), second.predict(features))


# --- the PyTorch rung -------------------------------------------------------


def test_missing_torch_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error that does not say how to fix itself is a dead end."""
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(RunError) as caught:
        build_torch_mlp("classification", seed=0)
    assert "pip install" in str(caught.value)


def test_torch_mlp_learns_a_separable_problem(
    classification: tuple[np.ndarray, np.ndarray],
) -> None:
    """The loop michi writes must actually train, not merely run."""
    pytest.importorskip("torch")
    features, labels = classification
    model = build_torch_mlp("classification", seed=0)
    model.set_params(epochs=40, hidden=(32,), patience=8)
    model.fit(features, labels)

    assert (model.predict(features) == labels).mean() > 0.7


def test_torch_mlp_predicts_the_original_labels(
    classification: tuple[np.ndarray, np.ndarray],
) -> None:
    """Predictions come back as the caller's classes, not internal indices."""
    pytest.importorskip("torch")
    features, _ = classification
    labels = np.array(["stay", "churn"])[
        (features[:, 0] + features[:, 1] > 0).astype(int)
    ]
    model = build_torch_mlp("classification", seed=0)
    model.set_params(epochs=20, hidden=(32,), patience=5)
    model.fit(features, labels)

    assert set(np.unique(model.predict(features))) <= {"stay", "churn"}


def test_torch_mlp_probabilities_are_a_distribution(
    classification: tuple[np.ndarray, np.ndarray],
) -> None:
    """`eval` calibrates and ranks with these; rows that do not sum to one lie."""
    pytest.importorskip("torch")
    features, labels = classification
    model = build_torch_mlp("classification", seed=0)
    model.set_params(epochs=20, hidden=(32,), patience=5)
    model.fit(features, labels)

    proba = model.predict_proba(features)
    assert proba.shape == (len(features), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_torch_regression_has_no_probabilities(
    regression: tuple[np.ndarray, np.ndarray],
) -> None:
    """Silently returning something would corrupt every downstream calibration."""
    pytest.importorskip("torch")
    features, values = regression
    model = build_torch_mlp("regression", seed=0)
    model.set_params(epochs=20, hidden=(32,), patience=5)
    model.fit(features, values)

    with pytest.raises(AttributeError):
        model.predict_proba(features)


def test_torch_regression_beats_predicting_the_mean(
    regression: tuple[np.ndarray, np.ndarray],
) -> None:
    """The floor every michi model is measured against."""
    pytest.importorskip("torch")
    features, values = regression
    model = build_torch_mlp("regression", seed=0)
    model.set_params(epochs=60, hidden=(32,), patience=10)
    model.fit(features, values)

    predicted = model.predict(features)
    assert predicted.shape == values.shape
    assert np.mean((predicted - values) ** 2) < np.var(values)


def test_torch_mlp_tolerates_holes_and_infinities(
    classification: tuple[np.ndarray, np.ndarray],
) -> None:
    """A NaN reaching a network makes every weight NaN on the first backward pass."""
    pytest.importorskip("torch")
    features, labels = classification
    features = features.copy()
    features[0, 0] = np.nan
    features[1, 1] = np.inf

    model = build_torch_mlp("classification", seed=0)
    model.set_params(epochs=20, hidden=(32,), patience=5)
    model.fit(features, labels)

    assert not np.isnan(model.predict_proba(features)).any()


def test_torch_mlp_exposes_its_hyperparameters_to_tune() -> None:
    """`tune --list-space` reads these; a hidden parameter cannot be searched."""
    pytest.importorskip("torch")
    params = build_torch_mlp("classification", seed=0).get_params()

    assert {"hidden", "dropout", "learning_rate", "epochs", "patience"} <= set(params)
