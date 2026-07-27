# `michi bench`

Train several models and find out which of them are *actually* different.

```bash
michi bench --list-models
michi bench data.csv --target churned --models linear,rf,hist-gbm
michi bench data.csv --target price --models ridge,lasso,rf --cv 10
michi bench data.csv --target y --models rf,xgb --report bench.html --open
```

## What makes this different

Any tool can train five models and sort them by score. The number that matters
is whether the gap between first and second means anything — and usually it
does not.

```
  Results  (ranked by balanced_accuracy)

  model     balanced_accuracy      95% interval   vs leader                fit
  ───────────────────────────────────────────────────────────────────────────
  tree                  0.716   0.6537 – 0.7792   leader                  0.0s
  knn                   0.715    0.613 – 0.8167   tied with leader (p=1)  0.0s
  linear                0.708   0.6278 – 0.7888   tied with leader (p=1)  0.1s
  rf                    0.694   0.6157 – 0.7723   tied with leader (p=1)  0.9s
  dummy                   0.5         0.5 – 0.5   worse (p=0.0123)        0.0s

  Verdict  tree scores highest, but knn, linear, rf are statistically
  indistinguishable from it at this sample size. Choosing between them on
  these numbers alone is not supported.
```

Four decisions produce that output:

**A dummy baseline is always added**, whether or not you asked for it. A
leaderboard without a floor invites the wrong conclusion.

**Differences are tested with the corrected resampled *t*-test**
(Nadeau & Bengio, 2003). Cross-validation folds share most of their training
data, so a naive paired *t*-test treats correlated evidence as independent and
calls noise significant. The correction inflates the variance by the
train/test ratio. It is deliberately conservative: michi would rather say
"cannot tell" than manufacture a winner.

**Many comparisons are Holm-adjusted.** Comparing every model against the
leader means many tests; without adjustment, one of them eventually looks
significant by chance.

**The verdict is a sentence.** "Not distinguishable at this sample size" is
what changes a decision — a p-value alone rarely does.

## Choosing models

```bash
michi bench --list-models
```

Prints the menu with a factual line about each. michi never picks for you and
never calls a model best. Everything trains **locally** — these are algorithms,
not downloads, and michi makes no network calls.

| Available in the base install | With `komichi[bench]` |
|---|---|
| `dummy`, `linear`, `ridge`, `lasso`, `tree`, `rf`, `extra-trees`, `hist-gbm`, `knn`, `svm`, `naive-bayes` | `xgb`, `lgbm`, `catboost` |

`hist-gbm` is sklearn's histogram gradient boosting — competitive with the
boosting libraries on many tabular problems, and it needs no extra.

## Column preparation

Models need numeric input, so `bench` must do *something* with missing values
and categories. What it does is printed on every run and written into every
manifest:

```
preparation: numeric: impute median · categorical: impute most_frequent +
onehot · standardise (scale-sensitive models only) — fitted inside each fold
```

Two things matter here. **It is stated**, not hidden — an assumption you can
read is not a decision michi made for you. And it is **fitted inside each
fold**, so imputers and encoders never see the fold they are scored on.
Fitting preprocessing on the whole dataset first is the most common way a
benchmark quietly leaks and reports numbers nobody can reproduce.

Override any of it:

```bash
michi bench data.csv --target y --impute mean --encode ordinal --no-scale
```

For real cleaning decisions — which columns to drop, how to treat outliers —
author a recipe with [`michi clean`](clean.md) and pass it:

```bash
michi bench data.csv --target y --recipe michi.recipe.yaml
```

The recipe's deterministic steps run once, up front; its fitted steps replace
michi's default preparation *inside each fold*. A recipe you wrote takes
precedence over michi's assumptions.

## Cross-validation

`--cv 5` by default, stratified for classification and shuffled with `--seed`.
If the rarest class has fewer members than the fold count, michi reduces the
folds to what the data supports rather than silently dropping stratification.

## Checks

| Check | Meaning |
|---|---|
| `below-baseline` | Nothing beat the dummy floor |
| `no-clear-winner` | Several models are statistically indistinguishable |
| `model-failed` | One model could not be trained; the rest still ran |

## `--explain` — reading your own numbers

A textbook can explain what a confidence interval is. What michi can explain,
and a textbook cannot, is why *your* interval is *that* wide and why *those
two* models could not be separated — using the folds that actually ran.

```
  Reading these numbers

    The dummy baseline scores 0.5 without looking at a single feature — it is
    what you get for free. linear scores 0.892, so the features are worth
    0.392 of balanced_accuracy here. That gap, not the headline score, is what
    the modelling bought.

    linear scored between 0.87 and 0.9338 across the 5 folds — a spread of
    0.06383 on the same data and the same settings, differing only in which
    120 rows were held out. That spread is the reason the interval has the
    width it does; it is measuring how much of the score is the fold rather
    than the model.

    rf scored 0.0141 below linear on average, but the two swapped places by
    ±0.0244 from fold to fold. The gap is smaller than the disagreement about
    the gap, which is what p=0.589 is reporting.

    Back-of-envelope: separating them at this effect size would take roughly
    3,200 rows, against the 600 here. That assumes the difference you measured
    is real and that error falls with the square root of the sample — both
    optimistic, so read it as an order of magnitude, not a target.
```

Every sentence there is a fact about the result or arithmetic on it. None of
them tells you what to do: "separating these would take roughly 3,200 rows" is
a calculation you can act on however you like; "you should collect more data"
would be a judgement, and michi does not make those. A test enforces the
distinction by scanning the notes for advice-shaped phrasing.

The sample-size figure is labelled back-of-envelope because it is one. It
assumes the measured difference is real and that error falls with √n — both
optimistic. It is an order of magnitude, not a target.

## Output

Every model gets its own run manifest in `runs/`, sharing a `group_id` so
`michi report` can regroup them. `--report bench.html` writes a self-contained
offline page.

## Options

| Option | Default | Purpose |
|---|---|---|
| `--target`, `-t` | *required* | Label column |
| `--models`, `-m` | `linear,rf,hist-gbm` | Models to compare |
| `--list-models` | — | Print the menu and exit |
| `--cv` | 5 | Cross-validation folds |
| `--task` | inferred | Force `classification` or `regression` |
| `--impute` | `median` | Numeric imputation |
| `--encode` | `onehot` | Categorical encoding |
| `--no-scale` | off | Never standardise |
| `--runs-dir` | `runs` | Where manifests are written |
| `--no-save` | off | Do not write manifests |
| `--report` | none | Write an HTML report |
| `--open` | off | Open the report in a browser |
| `--recipe` | none | Cleaning recipe to apply |
| `--seed` | 0 | Seed for folds and models |
| `--explain` | off | Explain this run's own numbers, and what each check means |

## A note on the intervals

The 95% interval shown is a *t*-interval across fold scores. It describes the
spread of folds, not true generalisation error, and is optimistic because
folds share training data. The significance test — which corrects for exactly
that overlap — is what the verdict relies on. Both are shown so you can see
the difference between them.
