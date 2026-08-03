# The Python API

Every verb is a thin layer over a function you can call directly. That matters
in a notebook — Kaggle, Colab, a research scratchpad — where shelling out to a
CLI is possible but unnatural.

```python
from pathlib import Path

from michi.core.io import load_table
from michi.inspection import profile_table
from michi.bench import run_benchmark

table = load_table(Path("train.csv"))
profile = profile_table(table, target="churned")
result = run_benchmark(table, target="churned", models=("linear", "rf"), folds=5)
```

**This surface is frozen** under [ADR-0005](adr/0005-freeze-the-expanded-surface.md):
each package's `__all__` is public API under semantic versioning. Anything not
in an `__all__` is private, whatever its visibility.

## Profiling

```python
from michi.inspection import profile_table
from michi.inspection.drift import compare_profiles

profile = profile_table(table, target="churned")
profile.n_rows, profile.n_columns
[f.summary for f in profile.findings_by_severity()]

drift = compare_profiles(baseline_profile, profile)
[f.summary for f in drift.findings]
```

`DatasetProfile.to_dict()` and `.from_dict()` round-trip through JSON, so a
profile written by `michi inspect --json` loads straight back.

## Comparing models

```python
from michi.bench import run_benchmark

result = run_benchmark(
    table,
    target="churned",
    models=("linear", "rf", "hist-gbm"),
    folds=5,
    group="customer_id",  # rows sharing an entity stay in one fold
    metric="rmsle",  # rank and test by the metric you care about
    balance=True,  # inverse-frequency class weights where supported
)

for item in result.results:
    print(item.name, item.primary.value, item.primary.ci_low, item.primary.ci_high)

for comparison in result.comparisons:
    print(comparison.model, comparison.verdict)
```

`result.leader` is the best model or `None` if all of them failed.
`result.manifests` holds one run manifest per model, ready to write.

## Recipes

```python
from michi.recipes import (
    Recipe,
    RecipeStep,
    apply_recipe,
    apply_deterministic,
    export_recipe,
    load_recipe,
    write_recipe,
)

recipe = Recipe(
    steps=(
        RecipeStep("drop", {"columns": ["id"]}, why="not a feature"),
        RecipeStep("log", {"columns": ["fare"]}),
    )
)
cleaned = apply_recipe(recipe, table.frame).frame
source = export_recipe(recipe)  # standalone Python, no michi import
write_recipe(recipe, Path("michi.recipe.yaml"))
```

`apply_recipe` runs everything; `apply_deterministic` runs only the steps that
cannot leak, which is what you want before splitting.

## Evaluating and predicting

```python
from michi.adapters import load_model
from michi.evaluation import evaluate_model
from michi.evaluation.importance import permutation_importance

model = load_model("model.joblib")  # or "mymodule:my_model"
manifest = evaluate_model(model, table, target="churned", importance=True)

manifest.metrics[0].value
manifest.details["confusion"]
manifest.details["importance"]

predictions = model.predict(features)  # any model with predict
```

## Tuning

```python
from michi.bench.tuning import search_space, tune_model

space = search_space("hist-gbm")  # printable, and replaceable
outcome = tune_model(
    features,
    labels,
    model="hist-gbm",
    task="classification",
    space=space,
    strategy="bayes",
    groups=group_values,
)
outcome.best_params, outcome.outer_score, outcome.optimism
```

`outer_score` is the honest number — folds the search never saw. `optimism` is
how much the search's own best overstated it, and it grows with the strength of
the optimiser.

## Adding your own metric

michi cannot ship every metric a competition will invent. Register one through
the `michi.metrics` entry point:

```toml
# pyproject.toml of your own package
[project.entry-points."michi.metrics"]
map_at_5 = "mypkg.metrics:map_at_5"
```

```python
# mypkg/metrics.py
def map_at_5(truth, prediction): ...


map_at_5.greater_is_better = True
```

Then `--metric map_at_5` works everywhere a metric name does, and because it
is an entry point rather than an import path, the metric travels with the
environment a run is reproduced in.

Model and loader plugins work the same way — see [the plugin guide](plugins.md).

## What stays out of the API

michi's CLI commands are *not* importable functions you should call: they
parse arguments, render to a terminal, and exit. Call the domain functions
above instead. The one-way dependency `core → domain → cli` is what keeps that
honest.
