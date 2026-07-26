"""Tests for the plugin surface.

Two properties matter: a working plugin extends michi, and a broken one is
isolated. The second is the harder promise, and the one that decides whether
michi stays supportable by one person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from michi.bench import ModelEntry, available_models, model_entry
from michi.cli.app import app
from michi.plugins import (
    ADAPTER_GROUP,
    MODEL_GROUP,
    PluginError,
    check_adapter,
    check_model_entry,
    discover,
    installed_plugins,
)

runner = CliRunner()


@dataclass(frozen=True)
class _FakeEntryPoint:
    """Stands in for an installed entry point."""

    name: str
    value: Any
    dist: Any = None

    def load(self) -> Any:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def _install(monkeypatch: pytest.MonkeyPatch, group: str, *points: Any) -> None:
    """Pretend the given entry points are installed."""
    discover.cache_clear()

    def _entry_points(group: str = "", **_: Any) -> tuple[Any, ...]:
        return points

    monkeypatch.setattr("importlib.metadata.entry_points", _entry_points)


def _sample_entry() -> ModelEntry:
    from sklearn.dummy import DummyClassifier, DummyRegressor

    def factory(task: str, seed: int) -> Any:
        return (
            DummyClassifier(strategy="most_frequent")
            if task == "classification"
            else DummyRegressor()
        )

    return ModelEntry(
        name="plugin-dummy",
        tasks=frozenset({"classification", "regression"}),
        summary="a model contributed by a plugin",
        factory=factory,
    )


# --- discovery -------------------------------------------------------------


def test_a_plugin_model_joins_the_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed model appears in `--list-models` and can be looked up."""
    _install(
        monkeypatch,
        MODEL_GROUP,
        _FakeEntryPoint(name="extra", value=lambda: (_sample_entry(),)),
    )
    names = {entry.name for entry in available_models()}
    assert "plugin-dummy" in names
    assert model_entry("plugin-dummy").summary


def test_a_broken_plugin_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin that cannot import must not stop michi from working."""
    _install(
        monkeypatch,
        MODEL_GROUP,
        _FakeEntryPoint(name="broken", value=ImportError("no such module")),
        _FakeEntryPoint(name="good", value=lambda: (_sample_entry(),)),
    )
    names = {entry.name for entry in available_models()}
    assert "rf" in names
    assert "plugin-dummy" in names


def test_a_broken_plugin_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failures are visible rather than silent."""
    _install(
        monkeypatch,
        MODEL_GROUP,
        _FakeEntryPoint(name="broken", value=ImportError("no such module")),
    )
    records = [item for item in installed_plugins() if not item.loaded]
    assert records
    assert "no such module" in (records[0].error or "")


def test_a_plugin_cannot_shadow_a_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    """`rf` always means what the documentation says."""
    shadow = ModelEntry(
        name="rf",
        tasks=frozenset({"classification"}),
        summary="an impostor",
        factory=lambda task, seed: None,
    )
    _install(
        monkeypatch,
        MODEL_GROUP,
        _FakeEntryPoint(name="shadow", value=lambda: (shadow,)),
    )
    assert model_entry("rf").summary != "an impostor"


def test_no_plugins_installed_is_normal() -> None:
    """The common case is no plugins at all, and it costs nothing."""
    discover.cache_clear()
    assert isinstance(installed_plugins(), tuple)


# --- the compatibility suite -----------------------------------------------


def test_a_valid_model_entry_passes_the_suite() -> None:
    """A well-formed contribution satisfies the published contract."""
    check_model_entry(_sample_entry())


def test_the_suite_rejects_a_missing_attribute() -> None:
    """An entry without a factory cannot be used."""

    class Incomplete:
        name = "x"
        tasks = frozenset({"classification"})
        summary = "no factory"

    with pytest.raises(PluginError, match="factory"):
        check_model_entry(Incomplete())


def test_the_suite_rejects_an_unknown_task() -> None:
    """Tasks michi does not have are caught before anything runs."""
    entry = ModelEntry(
        name="odd",
        tasks=frozenset({"clustering"}),
        summary="unsupported task",
        factory=lambda task, seed: None,
    )
    with pytest.raises(PluginError, match="unknown task"):
        check_model_entry(entry)


def test_the_suite_rejects_an_estimator_that_cannot_fit() -> None:
    """The contract is what michi actually calls: fit and predict."""

    class NotAnEstimator:
        def predict(self, features: Any) -> Any:
            return []

    entry = ModelEntry(
        name="broken",
        tasks=frozenset({"classification"}),
        summary="cannot fit",
        factory=lambda task, seed: NotAnEstimator(),
    )
    with pytest.raises(PluginError, match="fit"):
        check_model_entry(entry)


def test_the_suite_checks_adapters() -> None:
    """An adapter must recognise its own reference and return a model."""

    class Model:
        spec = "custom"

        def predict(self, features: Any) -> Any:
            return np.zeros(len(features))

    class Adapter:
        def handles(self, reference: str) -> bool:
            return reference.endswith(".custom")

        def load(self, reference: str) -> Any:
            return Model()

    check_adapter(Adapter(), "model.custom")


def test_the_suite_rejects_an_adapter_that_disowns_its_example() -> None:
    """An adapter that refuses its own example is misconfigured."""

    class Adapter:
        def handles(self, reference: str) -> bool:
            return False

        def load(self, reference: str) -> Any:
            return None

    with pytest.raises(PluginError, match="does not claim"):
        check_adapter(Adapter(), "model.custom")


def test_the_suite_rejects_a_model_without_predict() -> None:
    """michi calls predict; an adapter must return something that has it."""

    class Adapter:
        def handles(self, reference: str) -> bool:
            return True

        def load(self, reference: str) -> Any:
            return object()

    with pytest.raises(PluginError, match="predict"):
        check_adapter(Adapter(), "model.custom")


# --- the command -----------------------------------------------------------


def test_plugins_command_explains_the_groups_when_none_exist() -> None:
    """With nothing installed, michi documents how to contribute."""
    discover.cache_clear()
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    assert MODEL_GROUP in result.output
    assert ADAPTER_GROUP in result.output


def test_plugins_command_lists_what_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed plugins, working or not, are visible in one command."""
    _install(
        monkeypatch,
        MODEL_GROUP,
        _FakeEntryPoint(name="extra", value=lambda: (_sample_entry(),)),
    )
    result = runner.invoke(app, ["plugins"])
    assert "extra" in result.output


# --- isolation of real work ------------------------------------------------


def test_benchmarking_still_works_with_a_broken_plugin(
    tidy_csv: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin's mistake never becomes michi's failure."""
    _install(
        monkeypatch,
        MODEL_GROUP,
        _FakeEntryPoint(name="broken", value=RuntimeError("plugin exploded")),
    )
    result = runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear",
            "--cv",
            "3",
            "--no-save",
        ],
    )
    assert result.exit_code == 0


@pytest.fixture(autouse=True)
def _clear_plugin_cache() -> Any:
    """Discovery is cached per process; tests must not leak into each other."""
    discover.cache_clear()
    yield
    discover.cache_clear()
