"""Tests for model loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from michi.adapters import load_model
from michi.core.errors import ModelError

# --- pickle files ----------------------------------------------------------


def test_loads_a_pickled_estimator(
    classification_data: tuple[Path, Path],
) -> None:
    """An sklearn pipeline loads and reports its class and provenance."""
    model_path, _ = classification_data
    model = load_model(str(model_path))
    assert model.spec.loader == "joblib"
    assert model.spec.class_name == "Pipeline"
    assert model.spec.sha256 is not None


def test_records_hyperparameters(classification_data: tuple[Path, Path]) -> None:
    """Hyperparameters are captured so a run can be understood later."""
    model_path, _ = classification_data
    assert load_model(str(model_path)).spec.params != {}


def test_detects_probability_support(
    classification_data: tuple[Path, Path], regression_data: tuple[Path, Path]
) -> None:
    """michi knows whether a model can produce class probabilities."""
    classifier, _ = classification_data
    regressor, _ = regression_data
    assert load_model(str(classifier)).can_predict_proba is True
    assert load_model(str(regressor)).can_predict_proba is False


# --- the predict protocol --------------------------------------------------


def test_loads_any_object_with_predict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any object exposing predict works, with no framework support needed."""
    module = tmp_path / "custom_model.py"
    module.write_text(
        "class Model:\n"
        "    def predict(self, X):\n"
        "        return [0] * len(X)\n"
        "\n"
        "trained = Model()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    model = load_model("custom_model:trained")
    assert model.spec.loader == "protocol"
    assert model.spec.class_name == "Model"


def test_calls_a_factory_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference naming a factory is called to obtain the model."""
    module = tmp_path / "factory_model.py"
    module.write_text(
        "class Model:\n"
        "    def predict(self, X):\n"
        "        return [1] * len(X)\n"
        "\n"
        "def build():\n"
        "    return Model()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert load_model("factory_model:build").spec.class_name == "Model"


# --- failure modes ---------------------------------------------------------


def test_missing_model_file_is_actionable(tmp_path: Path) -> None:
    """A missing file explains both accepted forms of reference."""
    with pytest.raises(ModelError, match="mymodule:my_model"):
        load_model(str(tmp_path / "absent.pkl"))


def test_framework_formats_are_refused_with_a_route_forward(
    tmp_path: Path,
) -> None:
    """A .pt file is refused honestly, pointing at the predict protocol."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"not really a model")
    with pytest.raises(ModelError, match="predict"):
        load_model(str(path))


def test_object_without_predict_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """michi requires the one method it actually calls."""
    module = tmp_path / "bad_model.py"
    module.write_text("trained = 42\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ModelError, match="predict"):
        load_model("bad_model:trained")


def test_unimportable_module_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import failure names the module and where michi looked."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ModelError, match="could not import"):
        load_model("no_such_module:model")


def test_malformed_reference_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty attribute is a usage error with the expected form shown."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ModelError, match="mymodule:my_model"):
        load_model("module_only:")
