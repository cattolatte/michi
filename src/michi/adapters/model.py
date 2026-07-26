"""Loading a user's trained model.

Design Principles
-----------------
- **Two ways in, not ten.** michi loads sklearn-compatible estimators from
  pickle/joblib files, and *anything else* through a predict protocol
  (``module:object``). That covers PyTorch, TensorFlow, ONNX, and bespoke
  models today without michi shipping — and forever maintaining — a loader per
  framework.
- **No format sniffing beyond what is honest.** A ``.pt`` file or a SavedModel
  directory cannot be loaded without the code that defined the model. michi
  says so plainly instead of guessing and failing obscurely.
- **The model stays a black box.** michi calls ``predict`` (and
  ``predict_proba`` when present) and never inspects internals, so any model
  satisfying the protocol works identically.
- Loading executes user code by design; that risk is documented rather than
  silently mitigated.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from michi.core.errors import ModelError
from michi.core.hashing import hash_file
from michi.core.manifest import ModelSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

__all__ = ["LoadedModel", "load_model"]

_PICKLE_SUFFIXES = {".pkl", ".pickle", ".joblib", ".sav"}
_UNSUPPORTED_SUFFIXES = {
    ".pt": "PyTorch",
    ".pth": "PyTorch",
    ".onnx": "ONNX",
    ".h5": "Keras",
    ".keras": "Keras",
    ".pb": "TensorFlow",
    ".cbm": "CatBoost",
}


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A user's model behind michi's minimal prediction interface.

    Attributes
    ----------
    estimator
        The loaded object itself, called but never introspected.
    spec
        Provenance recorded in the run manifest.
    """

    estimator: Any
    spec: ModelSpec

    @property
    def can_predict_proba(self) -> bool:
        """Whether the model exposes calibrated-ish class probabilities."""
        return callable(getattr(self.estimator, "predict_proba", None))

    @property
    def classes(self) -> tuple[Any, ...] | None:
        """The model's class labels, when it declares them."""
        declared = getattr(self.estimator, "classes_", None)
        return None if declared is None else tuple(declared)

    def predict(self, features: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Predict labels or values for a feature frame.

        Raises
        ------
        ModelError
            If the model rejects the features, with the underlying complaint
            preserved — usually a column mismatch the user must resolve.
        """
        import numpy as np

        try:
            return np.asarray(self.estimator.predict(features))
        except Exception as err:  # third-party failure boundary
            raise ModelError(_prediction_hint(err, features)) from err

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray[Any, Any] | None:
        """Predict class probabilities, or ``None`` if unsupported."""
        import numpy as np

        if not self.can_predict_proba:
            return None
        try:
            return np.asarray(self.estimator.predict_proba(features))
        except Exception:  # third-party failure boundary
            # Probabilities are a bonus: a model that cannot produce them
            # still evaluates on every label-based metric.
            return None


def load_model(reference: str) -> LoadedModel:
    """Load a model from a pickle file or a ``module:object`` reference.

    Parameters
    ----------
    reference
        Either a path to a pickled sklearn-compatible estimator
        (``model.pkl``), or an import path to any object exposing
        ``predict`` (``mypackage.models:trained``).

    Returns
    -------
    LoadedModel
        The model behind michi's prediction interface.

    Raises
    ------
    ModelError
        If the reference cannot be resolved, or the object does not implement
        ``predict``.

    Notes
    -----
    Both paths execute code from the referenced artifact — unpickling and
    importing are equivalent to running it. Only load models you trust.
    """
    if ":" in reference and not _looks_like_windows_path(reference):
        model = _load_from_protocol(reference)
    else:
        model = _load_from_file(Path(reference))

    if not callable(getattr(model.estimator, "predict", None)):
        msg = (
            f"{reference} does not implement predict(X). michi evaluates any "
            "object with a predict method — wrap your model in a small class "
            "and pass it as 'mymodule:my_model'."
        )
        raise ModelError(msg)
    return model


def _load_from_file(path: Path) -> LoadedModel:
    """Load a pickled estimator, refusing formats that cannot be honest."""
    if not path.exists():
        msg = (
            f"no such model file: {path}. Pass a pickle/joblib file, or an "
            "import reference like 'mymodule:my_model'."
        )
        raise ModelError(msg)

    suffix = path.suffix.lower()
    if suffix in _UNSUPPORTED_SUFFIXES:
        framework = _UNSUPPORTED_SUFFIXES[suffix]
        msg = (
            f"michi does not load {framework} files directly: a {suffix} file "
            "cannot be reconstructed without the code that defined the model. "
            "Load it yourself and expose it as 'mymodule:my_model' — any "
            "object with predict(X) works."
        )
        raise ModelError(msg)
    if suffix not in _PICKLE_SUFFIXES:
        msg = (
            f"unrecognised model file {path.name!r}. michi reads "
            f"{', '.join(sorted(_PICKLE_SUFFIXES))} files, or any object "
            "referenced as 'mymodule:my_model'."
        )
        raise ModelError(msg)

    estimator = _unpickle(path)
    return LoadedModel(
        estimator=estimator,
        spec=ModelSpec(
            reference=str(path),
            loader="joblib",
            class_name=type(estimator).__name__,
            params=_describe_params(estimator),
            sha256=hash_file(path),
        ),
    )


def _unpickle(path: Path) -> Any:
    """Read a pickled object, preferring joblib's handling of large arrays."""
    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        pass
    except Exception as err:  # third-party failure boundary
        msg = f"could not load {path.name}: {err}"
        raise ModelError(msg) from err

    import pickle

    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as err:  # third-party failure boundary
        msg = f"could not unpickle {path.name}: {err}"
        raise ModelError(msg) from err


def _load_from_protocol(reference: str) -> LoadedModel:
    """Import ``module:object`` and adopt whatever it exposes."""
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        msg = (
            f"malformed model reference {reference!r}; expected "
            "'mymodule:my_model'"
        )
        raise ModelError(msg)

    _ensure_cwd_importable()
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        msg = (
            f"could not import {module_name!r}: {err}. michi imports from the "
            "current directory and your PYTHONPATH."
        )
        raise ModelError(msg) from err
    except Exception as err:  # third-party failure boundary
        msg = f"importing {module_name!r} failed: {err}"
        raise ModelError(msg) from err

    estimator: Any = module
    for part in attribute.split("."):
        try:
            estimator = getattr(estimator, part)
        except AttributeError as err:
            msg = f"{module_name!r} has no attribute {attribute!r}"
            raise ModelError(msg) from err

    # A reference may name the model itself or a zero-argument factory.
    if callable(estimator) and not callable(getattr(estimator, "predict", None)):
        try:
            estimator = estimator()
        except Exception as err:  # third-party failure boundary
            msg = f"calling {reference!r} to build a model failed: {err}"
            raise ModelError(msg) from err

    return LoadedModel(
        estimator=estimator,
        spec=ModelSpec(
            reference=reference,
            loader="protocol",
            class_name=type(estimator).__name__,
            params=_describe_params(estimator),
        ),
    )


def _ensure_cwd_importable() -> None:
    """Put the working directory on the import path, as ``python -m`` does."""
    import sys

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _describe_params(estimator: Any) -> dict[str, str]:
    """Record an estimator's hyperparameters, stringified for the manifest."""
    getter = getattr(estimator, "get_params", None)
    if not callable(getter):
        return {}
    try:
        params = getter(deep=False)
    except Exception:  # third-party failure boundary
        return {}
    if not isinstance(params, dict):
        return {}
    return {str(key): repr(value) for key, value in sorted(params.items())}


def _prediction_hint(error: Exception, features: pd.DataFrame) -> str:
    """Turn a prediction failure into something the user can act on."""
    message = str(error)
    if "feature names" in message.lower() or "columns" in message.lower():
        return (
            f"the model rejected these features: {message}. michi passes every "
            f"column except the target ({features.shape[1]} columns). If the "
            "model expects preprocessed input, evaluate a pipeline that "
            "includes the preprocessing, or select columns with --features."
        )
    return f"prediction failed: {message}"


def _looks_like_windows_path(reference: str) -> bool:
    """Whether a colon is a Windows drive letter rather than a separator."""
    return len(reference) > 1 and reference[1] == ":" and reference[0].isalpha()
