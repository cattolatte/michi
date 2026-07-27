# `michi tune` · `fit` · `predict`

The last mile: search hyperparameters, train the model you chose, and produce
predictions for data that has no answers.

```bash
michi tune data.csv --target churned --model hist-gbm --save-params best.yaml
michi fit  data.csv --target churned --model hist-gbm --params best.yaml -o model.joblib
michi predict model.joblib test.csv --id id --proba -o submission.csv
```

## `michi tune` — hyperparameter search

### The space is printable

```bash
michi tune --model rf --list-space
```

```
 道  michi tune  ·  rf  ·  space search

  parameter          values
  ───────────────────────────────────
  max_depth          8, 16, None
  max_features       'sqrt', 0.5, 1.0
  min_samples_leaf   1, 5, 20
  n_estimators       200, 500

  54 combinations  ·  michi's built-in space
```

Nothing is searched that you cannot read first. Replace the space entirely
with your own:

```yaml
# my_space.yaml — bare names work; michi adds its own pipeline prefix
max_depth: [2, 4, 8, 16]
min_samples_leaf: [1, 10]
```

```bash
michi tune data.csv --target churned --model rf --space my_space.yaml
```

Built-in spaces exist for `ridge`, `lasso`, `linear`, `tree`, `rf`,
`extra-trees`, `hist-gbm`, `knn`, `svm`, `xgb`, `lgbm`, and `catboost`. They
are deliberately modest: a space too large for the search to cover turns
tuning into a lottery whose result does not replicate.

### Strategies

| `--strategy` | What it does |
|---|---|
| `random` (default) | Samples `--candidates` configurations. Best value for a fixed budget on most spaces. |
| `halving` | Successive halving — many configurations on little data, survivors on more. |
| `grid` | Every combination. Honest, and expensive. |

### The number it reports is not the search's own

```
                           balanced_accuracy
  ──────────────────────────────────────────
  tuned (held-out folds)              0.8891
  defaults (same folds)                0.869
  search's own best                   0.8943

  Verdict  tuning gained 0.02019 of balanced_accuracy over the defaults,
  measured on folds the search never saw.
  The search's own best score was 0.00519 better than the held-out result.
  That gap is what reporting an inner score as performance would have hidden.
```

Three numbers, and the order matters.

**`tuned (held-out folds)`** is the honest one. The search runs *inside* each
outer training fold; this score comes from data no configuration was ever
chosen on.

**`defaults (same folds)`** is the same model untuned on the same folds —
otherwise "tuning gained 0.02" has nothing to be 0.02 against.

**`search's own best`** is what most tools print as the result. It is the
maximum over many draws on the data that picked them, so it is optimistic by
construction. michi shows it *next to* the honest number rather than instead
of it, because the gap is the lesson.

Reporting an inner search score as performance is the second most common
silent leak in tabular ML, after target encoding.

### Options

| Option | Default | Purpose |
|---|---|---|
| `--model`, `-m` | `hist-gbm` | Model to tune |
| `--strategy` | `random` | `random`, `halving`, or `grid` |
| `--candidates` | 30 | Configurations to try (random only) |
| `--space` | built-in | YAML search space replacing michi's |
| `--list-space` | off | Print the space and exit |
| `--save-params` | none | Write the winner as YAML for `fit --params` |
| `--cv` / `--inner-cv` | 5 / 3 | Outer and inner folds |
| `--recipe` | none | Cleaning recipe to apply first |
| `--seed` | 0 | Seed for sampling and folds |

## `michi fit` — train and save

```bash
michi fit data.csv --target churned --model hist-gbm --recipe michi.recipe.yaml \
  --params best.yaml -o model.joblib
```

Trains one model on **every row you give it** and writes a joblib file.

**It reports no accuracy, on purpose.** A score measured on the rows a model
trained on is the most confidently wrong number a tool can print. `michi
bench` compares models honestly; `michi eval` scores one against held-out
data. `fit` exists to produce the artifact, nothing more.

## `michi predict` — the file you actually submit

```bash
michi predict model.joblib test.csv --id customer_id --proba -o submission.csv
```

```csv
id,prediction,proba_0,proba_1
0,1,0.0112,0.9888
1,0,0.8517,0.1483
```

Needs **no label column** — that is the whole point, and the one thing `eval`
cannot do. `--recipe` applies the same cleaning the model was trained under.
`--drop-target` removes a label column that a held-out file still carries.

### Any model, including deep learning

`predict` accepts anything `eval` does:

```bash
michi predict model.joblib test.csv -o preds.csv       # sklearn pickle/joblib
michi predict mynet:model test.csv -o preds.csv        # PyTorch, TF, ONNX, yours
```

The second form imports `mynet` and uses its `model` object. michi calls
`predict` and `predict_proba` and never inspects what is behind them, so a
torch module wrapped in a class with a `predict` method works identically to a
random forest — with no per-framework loader for michi to maintain and get
wrong.

```python
# mynet.py
import torch


class Net:
    classes_ = [0, 1]

    def __init__(self, weights="net.pt"):
        self.module = torch.load(weights)
        self.module.eval()

    def predict(self, frame):
        with torch.no_grad():
            x = torch.tensor(frame.to_numpy(dtype="float32"))
            return self.module(x).argmax(dim=1).numpy()


model = Net()
```

### Options

| Option | Default | Purpose |
|---|---|---|
| `--out`, `-o` | `predictions.csv` | Where to write. `.csv`, `.tsv`, `.parquet` |
| `--id` | none | Column carried through beside the prediction |
| `--label` | `prediction` | Name of the prediction column |
| `--proba` | off | Write class probabilities where available |
| `--recipe` | none | Cleaning recipe to apply first |
| `--drop-target` | from `michi.toml` | Label column to remove if present |

## The whole loop

```bash
michi inspect  data/train.csv --target target --explain
michi clean    data/train.csv --target target -o recipe.yaml
michi bench    data/train.csv --target target --recipe recipe.yaml \
               --models linear,rf,hist-gbm,xgb --explain
michi tune     data/train.csv --target target --recipe recipe.yaml \
               --model hist-gbm --save-params best.yaml
michi fit      data/train.csv --target target --recipe recipe.yaml \
               --model hist-gbm --params best.yaml -o model.joblib
michi predict  model.joblib data/test.csv --recipe recipe.yaml \
               --id id --proba -o submission.csv
```

Every step is optional and every step stands alone. `michi path` shows this
map from inside the console; `michi walk` asks its way through it.
