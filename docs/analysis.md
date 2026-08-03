# `michi threshold` · `errors` · `split`

Three questions a score does not answer: where the line should sit, which rows
made up the misses, and whether the split that produced the score was honest.

## `michi threshold` — the cutoff nobody chose

A classifier emits a probability. Turning it into a label needs a number, and
almost every tool picks **0.5 in silence**. On imbalanced data, or when a miss
costs more than a false alarm, that is simply the wrong number and nothing
says so.

```bash
michi threshold model.pkl data.csv --target churned --cost fn=10,fp=1 --objective cost
```

```
      cutoff   precision   recall      F1   flagged   missed   false alarms    cost
  ─────────────────────────────────────────────────────────────────────────────────
        0.05       0.668    1.000   0.801       355        0            118     118
  ▸     0.16       0.774    0.996   0.871       305        1             69      79
        0.39       0.920    0.966   0.942       249        8             20     100
        0.50       0.936    0.924   0.930       234       18             15     195
        0.95       1.000    0.540   0.701       128      109              0   1,090
```

On that data the best cutoff is **0.16, not 0.50** — and 0.50 costs 195
against 79, more than twice as much.

michi marks the best cutoff **for an objective you named**: `f1`,
`balanced_accuracy`, `precision`, `recall`, or `cost`. It never says which
objective to want. `--cost fn=10,fp=1` states that a missed positive is ten
times worse than a false alarm; michi has no way to know whether that is true,
so it has no default beyond weighing them equally.

| Option | Default | Purpose |
|---|---|---|
| `--objective` | `f1` | What to mark. `cost` requires `--cost` |
| `--cost` | equal | Relative error costs, e.g. `fn=10,fp=1` |
| `--positive` | second class | Which class counts as positive |
| `--steps` | 19 | How many cutoffs to evaluate |
| `--recipe` | none | Cleaning recipe to apply first |

## `michi errors` — the rows behind the score

After `eval` says 0.88, the next question is always which rows made up the
0.12. Every practitioner then writes the same twenty lines of pandas.

```bash
michi errors model.pkl data.csv --target churned --show 5 --out mistakes.csv
```

```
  33 of 600 rows wrong (5.5%)  ·  ordered by how sure the model was

  sure   predicted   age   salary      region   fare
   84%   0           30    —           south    11.7
   82%   1           59    4.562e+04   west     1e+04

  Where they concentrate

    · region = 'west': 10.7% wrong (6 of 56), against 5.5% overall
```

**Ordered by confidence**, because that ordering separates the two kinds of
mistake. A confident error is a mislabelled row, a leak, or a region the
features do not describe. An unsure one is the model behaving correctly at the
edge — and there is nothing to fix there.

Patterns are **described, never diagnosed**: michi can see that the west
region fails at twice the base rate and cannot see why. `--out` writes every
mistake, because the useful next step happens somewhere that is not a
terminal. A regression target has no "wrong", only "far", so rows are ranked
by residual instead.

## A loaded model runs code

`predict`, `threshold`, and `errors` all load a model you name, and loading a
pickle **executes whatever is inside it**. That is not a michi weakness — it
is how `pickle` and `joblib` work, and why `module:object` exists as the
alternative — but it means the same caution applies as to running any script:
evaluate models you trust, or ones you produced yourself with `michi fit`.

michi never downloads a model, and never loads one you did not name.

## `michi split` — holding data out honestly

```bash
michi split data.csv --target churned --group customer_id
michi split data.csv --time signup_date --test-size 0.25
```

A random split is right until the rows are not independent, and then it is
confidently wrong.

| Strategy | When |
|---|---|
| `stratified` | **Default** for a classification target — an unstratified split on imbalanced data can starve a fold |
| `group` | Rows share an entity. Four of a customer's rows in training and one in test is memory, not generalisation |
| `time` | A series. A test set drawn from *before* the training rows asks the model to predict the past |
| `random` | Everything else — and it says plainly what it cannot promise |

The summary states the property the split was chosen to give, then **verifies
it**:

```
  No value of customer_id appears on both sides — the leak a random split
  would have caused is absent.
```

If a group *does* span both sides it says so loudly, rather than printing a
guarantee that did not hold.

### The same grouping belongs in cross-validation

Splitting correctly and then cross-validating carelessly puts the leak back.
Every verb that cross-validates takes `--group`:

```bash
michi bench    data.csv --target churned --group customer_id
michi tune     data.csv --target churned --group customer_id
michi ensemble data.csv --target churned --group customer_id
```

On a dataset where the label is a property of the customer, omitting it
overstated balanced accuracy by **26 points** — 0.836 against an honest 0.580.
The grouping column is never passed to the model as a feature; handing it the
entity id would defeat the point.

| Option | Default | Purpose |
|---|---|---|
| `--test-size` | 0.2 | Fraction held out |
| `--strategy` | `auto` | `auto`, `random`, `stratified`, `group`, `time` |
| `--group` / `--time` | none | The column each strategy needs |
| `--train` / `--test` | `train.csv` / `test.csv` | Where to write |
