# `michi eval`

Evaluate a model michi never saw trained, against a dataset you already have.

```bash
michi eval model.pkl test.csv --target churned
michi eval model.pkl test.csv --target churned --explain
michi eval mypackage.models:trained test.parquet --target price
michi eval model.pkl test.csv --target y --fail-under f1=0.85
```

Every run writes a **run manifest** to `runs/` — the durable record of what
was measured, on which bytes, with which model, in which environment.

## What it reports

**Metrics with confidence intervals.** Every headline metric comes with a 95%
bootstrap interval, because the difference between 0.91 and 0.89 means nothing
when the interval spans 0.06. Classification is headed by *balanced accuracy*
(which does not flatter a model that learned the majority class); regression by
*RMSE* (in the units of your target, so a domain expert can judge it).

**Baselines, always.** Trivial models run on the same rows every time —
most-frequent and stratified for classification, mean and median for
regression. A score is meaningless without knowing what guessing achieves.

**Where the model fails, not just how often.** A confusion matrix, per-slice
scores for every low-cardinality column, and a calibration curve for models
that produce probabilities.

**Checks** — the questions a careful reviewer asks:

| Check | Meaning |
|---|---|
| `below-baseline` | The model does not beat guessing |
| `beats-baseline` | It does, and by how much |
| `suspiciously-perfect` | A near-perfect score — usually leakage, not skill |
| `single-class-predictions` | Every prediction is the same class |
| `wide-interval` | The interval is too wide to support fine comparisons |
| `miscalibrated` | Predicted probabilities do not match observed frequencies |
| `slice-gap` | A large performance gap between subgroups |
| `small-evaluation-set` | Too few rows for the metrics to mean much |

`--explain` prints what each check means and the options practitioners choose
between — never a recommendation.

## Which models work

**sklearn-compatible pickles.** `model.pkl`, `model.joblib`. If the pickle is
a `Pipeline` that includes preprocessing, michi passes your raw columns
straight through it, which is the case that just works.

**Anything else, through the predict protocol.** Point michi at any Python
object with a `predict(X)` method:

```python
# mypackage/models.py
import torch


class Wrapper:
    def __init__(self):
        self.net = torch.load("net.pt", weights_only=False).eval()

    def predict(self, X):
        with torch.no_grad():
            return (
                self.net(torch.tensor(X.values, dtype=torch.float32)).argmax(1).numpy()
            )


trained = Wrapper()
```

```bash
michi eval mypackage.models:trained test.csv --target label
```

Three lines of your code covers PyTorch, TensorFlow, ONNX, and anything
bespoke — which is why michi ships no per-framework loaders. A `.pt` file or a
SavedModel cannot be reconstructed without the code that defined it, so michi
refuses those directly rather than guessing and failing obscurely.

If the reference names a zero-argument factory, michi calls it.

!!! warning "Loading a model runs its code"
    Unpickling and importing both execute code from the artifact. Only
    evaluate models you trust — the same caution that applies to
    `joblib.load` anywhere.

## Feature handling

By default michi passes **every column except the target** to the model. If
your model expects a subset, or a specific order:

```bash
michi eval model.pkl test.csv --target y --features age,income,region
```

If the model expects preprocessed input, evaluate a pipeline that includes the
preprocessing, or apply a recipe first:

```bash
michi eval model.pkl test.csv --target y --recipe michi.recipe.yaml
```

Only the recipe's *deterministic* steps run — dropping, casting, clipping.
Imputers and encoders in a recipe were fitted for training, and re-fitting
them on the evaluation set would quietly change what is being measured.

## Slices

Low-cardinality columns are scored separately by default. To choose them:

```bash
michi eval model.pkl test.csv --target y --slice region,plan_tier
```

Slices are how you discover that a model with 80% accuracy overall is at 47%
for one segment — a gap an aggregate score hides completely.

## In CI

```bash
michi eval model.pkl test.csv --target y --fail-under balanced_accuracy=0.75
```

The direction is taken from the metric, so `--fail-under rmse=3.0` passes when
RMSE is *at most* 3.0. Exit codes: `0` gate passed, `1` gate failed, `2` the
model or data could not be read.

## Options

| Option | Default | Purpose |
|---|---|---|
| `--target`, `-t` | *required* | Label column |
| `--task` | inferred | Force `classification` or `regression` |
| `--features` | all but target | Columns to pass to the model |
| `--slice` | low-cardinality columns | Columns to score subgroups over |
| `--recipe` | none | Cleaning recipe to apply before evaluating |
| `--runs-dir` | `runs` | Where manifests are written |
| `--no-save` | off | Do not write a manifest |
| `--json` | none | Also write the manifest here |
| `--explain` | off | Meaning and options for each check |
| `--bootstrap` | 1000 | Resamples for intervals; `0` disables |
| `--sample` / `--full` | 200000 | Sampling for large files |
| `--seed` | 0 | Seed for sampling and resampling |
| `--fail-under` | none | CI gate, e.g. `f1=0.85` |

## The run manifest

```json
{
  "schema_version": "1.0",
  "run_id": "20260726T175344Z-6f7b2f58",
  "kind": "eval",
  "dataset": { "sha256": "ce552926…", "total_rows": 490 },
  "target": "churned",
  "task": "classification",
  "model": { "reference": "model.pkl", "class_name": "Pipeline", "sha256": "…" },
  "metrics": [
    { "name": "balanced_accuracy", "value": 0.655,
      "ci_low": 0.610, "ci_high": 0.700, "greater_is_better": true }
  ],
  "baselines": { "most_frequent": [ … ], "stratified": [ … ] },
  "checks": [ … ],
  "environment": { "python": "3.13.9", "packages": { "scikit-learn": "1.7.0" } }
}
```

Manifests are the input to `michi report` (v0.3). They are plain JSON with a
versioned schema — readable, diffable, and interpretable without michi
installed.
