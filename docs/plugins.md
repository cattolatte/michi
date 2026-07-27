# Plugins

michi has two extension points. Both were opened at v0.8, not before, because
the interfaces behind them had by then survived six milestones and a dozen
concrete implementations without changing shape. An interface with one
implementation is a guess.

```bash
michi plugins        # what is installed, working or not
```

## `michi.models` — add an algorithm

Contribute entries to the `bench` catalogue:

```python
# my_plugin/__init__.py
from michi.bench import ModelEntry


def _build(task: str, seed: int):
    from my_library import MyClassifier, MyRegressor

    return (
        MyClassifier(random_state=seed)
        if task == "classification"
        else MyRegressor(random_state=seed)
    )


def models():
    return (
        ModelEntry(
            name="my-model",
            tasks=frozenset({"classification", "regression"}),
            summary="what it is, factually — never a claim that it is best",
            factory=_build,
            needs_scaling=False,
        ),
    )
```

```toml
# pyproject.toml
[project.entry-points."michi.models"]
my_plugin = "my_plugin:models"
```

Your model then appears in `michi bench --list-models`, `show models`, and can
be named in a sweep like any built-in.

## `michi.adapters` — add a way to load a model

michi deliberately ships no per-framework loaders: a `.pt` file cannot be
reconstructed without the code that defined it. If you *do* have that context —
an ONNX runtime, a framework convention your team follows — an adapter is where
it belongs.

An adapter is any object with `handles(reference)` and `load(reference)`:

```python
class OnnxAdapter:
    def handles(self, reference: str) -> bool:
        return reference.endswith(".onnx")

    def load(self, reference: str):
        import onnxruntime

        from michi.adapters import LoadedModel
        from michi.core.manifest import ModelSpec

        session = onnxruntime.InferenceSession(reference)

        class Wrapper:
            def predict(self, features):
                name = session.get_inputs()[0].name
                return session.run(None, {name: features.to_numpy("float32")})[0]

        return LoadedModel(
            estimator=Wrapper(),
            spec=ModelSpec(reference=reference, loader="onnx", class_name="Onnx"),
        )
```

```toml
[project.entry-points."michi.adapters"]
onnx = "my_plugin:OnnxAdapter"
```

Adapters are offered every reference before michi's built-in paths, so yours
can accept something michi would otherwise refuse.

## Test yourself

michi publishes the contract so that **you** verify your plugin, not michi's
maintainer. This is the only way a plugin ecosystem stays affordable for one
person to support.

```python
from michi.plugins import check_adapter, check_model_entry

from my_plugin import OnnxAdapter, models


def test_michi_model_contract():
    for entry in models():
        check_model_entry(entry)


def test_michi_adapter_contract():
    check_adapter(OnnxAdapter(), "fixtures/model.onnx")
```

The suite checks exactly what michi calls: that a model entry is well-formed
and its estimator can fit and predict, and that an adapter recognises its own
reference and returns something with `predict`.

## Rules michi enforces

**A broken plugin is skipped, never fatal.** An entry point that fails to
import is reported by `michi plugins` and ignored. Your users' benchmarks keep
running.

**Built-ins win ties.** A plugin cannot shadow `rf` or any other documented
name, because a user reading `--list-models` must be able to trust the
documentation.

**Summaries are factual.** The catalogue describes what a model *is*, never
that it is best — the same rule michi's own entries follow.

## Why only two

Recipe operations, report sections, and metric definitions are all plausible
extension points, and all closed. They stay closed until something real needs
them, because a published interface is a promise, and a promise made on a
guess is expensive to keep.
