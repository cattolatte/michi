"""Hyperparameter overrides: the defaults are a starting point, not a ceiling.

Every catalogue model carries defaults, and every one of them is meant to be
overridable — michi picks mechanics, never decisions. These tests defend both
halves: that an override actually reaches the estimator, and that a mistyped
one says so in michi's voice rather than scikit-learn's.
"""

from __future__ import annotations

import pytest
from sklearn.linear_model import Ridge

from michi.bench import apply_params
from michi.core.errors import RunError

# --- the override reaching the model ----------------------------------------


def test_a_parameter_is_actually_set() -> None:
    """The whole point: what the user asked for is what the estimator carries."""
    assert apply_params(Ridge(), {"alpha": 2.5}).alpha == 2.5


def test_several_parameters_apply_together() -> None:
    """Users pass a file, not a flag, so one at a time is not the real case."""
    model = apply_params(Ridge(), {"alpha": 0.5, "fit_intercept": False})
    assert model.alpha == 0.5
    assert model.fit_intercept is False


def test_an_empty_mapping_changes_nothing() -> None:
    """`--params` is optional everywhere it exists, so absent must be a no-op."""
    assert apply_params(Ridge(alpha=3.0), {}).alpha == 3.0


def test_the_estimator_comes_back_for_chaining() -> None:
    """Callers build and configure in one expression."""
    model = Ridge()
    assert apply_params(model, {"alpha": 1.5}) is model


# --- the override failing readably ------------------------------------------


def test_an_unknown_parameter_names_itself() -> None:
    """This shipped as a raw traceback, which is not an error message.

    Misremembering `epochs` against `max_iter` across michi's two neural
    models is close to certain, so this is the first thing a user hits.
    """
    with pytest.raises(RunError) as caught:
        apply_params(Ridge(), {"alphaa": 1.0})
    assert "alphaa" in str(caught.value)


def test_an_unknown_parameter_lists_what_would_have_worked() -> None:
    """Naming the mistake without naming the fix leaves the user guessing."""
    with pytest.raises(RunError) as caught:
        apply_params(Ridge(), {"alphaa": 1.0})
    assert "alpha" in str(caught.value)
    assert "fit_intercept" in str(caught.value)


def test_several_unknown_parameters_are_reported_at_once() -> None:
    """Fixing a file one error per run is a slow way to spend an afternoon."""
    with pytest.raises(RunError) as caught:
        apply_params(Ridge(), {"alphaa": 1.0, "intercept": True})
    message = str(caught.value)
    assert "alphaa" in message and "intercept" in message


def test_a_rejected_value_is_wrapped_rather_than_raised_raw() -> None:
    """A known name with an impossible value must not escape as a raw error.

    scikit-learn's own estimators mostly defer value checking to `fit`, so
    this branch needs an estimator that refuses at set time — a plugin model
    is free to, and michi's contract is that third-party failures arrive
    wrapped whoever raised them.
    """

    class Picky:
        def get_params(self, deep: bool = True) -> dict[str, object]:
            return {"width": 1}

        def set_params(self, **params: object) -> None:
            raise ValueError("width must be positive")

    with pytest.raises(RunError) as caught:
        apply_params(Picky(), {"width": -1})
    assert "width must be positive" in str(caught.value)
    assert caught.value.__cause__ is not None  # chained, per the repo's rule
