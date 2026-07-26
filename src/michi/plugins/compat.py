"""The compatibility suite plugin authors run in their own CI.

Design Principles
-----------------
- **Authors test themselves.** A published contract suite means michi's
  maintainer is not the integration point for every plugin, which is the only
  way a plugin ecosystem stays affordable for one person to support.
- **The contract is what michi actually calls**, nothing more: a model entry
  must build an estimator that fits and predicts; an adapter must recognise
  its own references and return something with ``predict``.
- Failures raise :class:`~michi.plugins.registry.PluginError` with a message
  naming what the plugin got wrong.
"""

from __future__ import annotations

from typing import Any

from michi.plugins.registry import PluginError

__all__ = ["check_adapter", "check_model_entry"]


def check_model_entry(entry: Any) -> None:
    """Verify a model catalogue entry satisfies michi's contract.

    Call this from your plugin's test suite::

        from michi.plugins.compat import check_model_entry
        from my_plugin import my_entry

        def test_michi_compatibility():
            check_model_entry(my_entry)

    Raises
    ------
    PluginError
        If the entry is malformed, or its estimator cannot fit and predict on
        a trivial dataset.
    """
    import numpy as np
    import pandas as pd

    for attribute in ("name", "tasks", "summary", "factory"):
        if not hasattr(entry, attribute):
            msg = f"a model entry needs a {attribute!r} attribute"
            raise PluginError(msg)

    if not isinstance(entry.name, str) or not entry.name:
        msg = "a model entry needs a non-empty string name"
        raise PluginError(msg)
    if not entry.tasks:
        msg = f"{entry.name}: declare at least one task"
        raise PluginError(msg)

    unknown = set(entry.tasks) - {"classification", "regression"}
    if unknown:
        msg = (
            f"{entry.name}: unknown task(s) {', '.join(sorted(unknown))}; "
            "expected classification and/or regression"
        )
        raise PluginError(msg)

    rows = 40
    features = pd.DataFrame(
        {"a": np.arange(rows, dtype=float), "b": np.arange(rows) % 4}
    )
    for task in sorted(entry.tasks):
        labels = (
            np.arange(rows) % 2
            if task == "classification"
            else np.arange(rows, dtype=float)
        )
        try:
            estimator = entry.factory(task, 0)
        except Exception as err:
            msg = f"{entry.name}: factory({task!r}, 0) raised {err}"
            raise PluginError(msg) from err

        for method in ("fit", "predict"):
            if not callable(getattr(estimator, method, None)):
                msg = f"{entry.name}: estimator has no {method}() method"
                raise PluginError(msg)

        try:
            estimator.fit(features, labels)
            predictions = np.asarray(estimator.predict(features))
        except Exception as err:
            msg = f"{entry.name}: could not fit and predict for {task}: {err}"
            raise PluginError(msg) from err

        if predictions.shape[0] != rows:
            msg = (
                f"{entry.name}: predict returned {predictions.shape[0]} values "
                f"for {rows} rows"
            )
            raise PluginError(msg)


def check_adapter(adapter: Any, reference: str) -> None:
    """Verify a model-loading adapter satisfies michi's contract.

    An adapter is any object with ``handles(reference) -> bool`` and
    ``load(reference) -> LoadedModel``. Call this from your plugin's tests
    with a reference your adapter should accept.

    Raises
    ------
    PluginError
        If the adapter does not recognise its own reference, or returns
        something michi cannot call ``predict`` on.
    """
    for method in ("handles", "load"):
        if not callable(getattr(adapter, method, None)):
            msg = f"an adapter needs a {method}() method"
            raise PluginError(msg)

    try:
        recognised = bool(adapter.handles(reference))
    except Exception as err:
        msg = f"handles({reference!r}) raised {err}"
        raise PluginError(msg) from err
    if not recognised:
        msg = f"adapter does not claim to handle its own example {reference!r}"
        raise PluginError(msg)

    try:
        model = adapter.load(reference)
    except Exception as err:
        msg = f"load({reference!r}) raised {err}"
        raise PluginError(msg) from err

    if not callable(getattr(model, "predict", None)):
        msg = "the loaded model has no predict() method"
        raise PluginError(msg)
    if not hasattr(model, "spec"):
        msg = "a loaded model must carry a `spec` describing its provenance"
        raise PluginError(msg)
