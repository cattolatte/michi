"""Tests for the michi error hierarchy."""

from __future__ import annotations

import pytest

from michi.core import (
    DataError,
    MichiError,
    ModelError,
    RecipeError,
    ReportError,
    RunError,
)

# --- hierarchy -------------------------------------------------------------


def test_all_domain_errors_subclass_michi_error() -> None:
    """Every domain error is catchable as MichiError."""
    for exc_type in (DataError, ModelError, RecipeError, RunError, ReportError):
        assert issubclass(exc_type, MichiError)


def test_michi_error_subclasses_exception() -> None:
    """MichiError behaves like a normal exception."""
    with pytest.raises(MichiError, match="boom"):
        raise DataError("boom")


# --- chaining --------------------------------------------------------------


def test_errors_preserve_cause_chain() -> None:
    """Wrapped third-party failures keep their original cause."""
    cause = ValueError("bad csv")
    try:
        raise DataError("could not parse data.csv") from cause
    except DataError as err:
        assert err.__cause__ is cause
