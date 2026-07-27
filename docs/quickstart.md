# Quickstart

Fifteen minutes, one messy CSV, and no commitment. Nothing here asks you to
restructure a project, adopt a template, or import anything.

```bash
pip install michi-ml
```

(The distribution is `michi-ml`; the command is `michi`.)

## 1. Look at the data

```bash
michi inspect data/customers.csv --target purchased
```

```
 道  michi inspect  ·  customers.csv

  120 rows × 13 columns  ·  15.3% of cells missing  ·  0 duplicate rows
  target purchased
  sha256 8b0f2ad7de62  ·  7.3 KB

  column         kind          missing   unique   summary
  ────────────────────────────────────────────────────────────────────────
  age            numeric             —       45   mean 41.625 · range 20–64
  salary         numeric         11.7%      106   mean 38,169
  cabin          categorical     87.5%       15   top: C0 (1), C8 (1)
  notes          empty          100.0%        0
  fare           numeric             —       11   mean 259 · skew +6.16
  …

  Findings (15)

  high    notes                  every value is missing
  high    country                only one distinct value (JP)
  high    cabin                  87.5% missing (105 of 120)
  high    outcome_code           each of its 2 values maps to exactly one
                                 'purchased' class
  warn    purchased              smallest class 8.3% vs largest 91.7%
  warn    age, age_months        correlation +1.000
  …
```

That fourth finding is the one worth pausing on. `outcome_code` predicts the
label perfectly — almost always a value recorded *after* the outcome, which
will not exist when the model is used. Catching it now saves a week later.

Not sure what a finding means?

```bash
michi inspect data/customers.csv --target purchased --explain
```

Every finding gets what it means and the options practitioners choose between
— never a recommendation. michi describes; you decide.

## 2. Decide what to do about it

```bash
michi clean data/customers.csv --target purchased
```

michi walks the *findings*, grouped, with "leave them as they are" always
present and always the default:

```
  [2/6]  4 columns are some missing values
         salary 11.7% missing (14 of 120); age 3.2% missing (4 of 120)

  What would you like to do?
  ❯ impute the median for 3 numeric column(s)
    impute the mean for 3 numeric column(s)
    drop the affected rows
    drop the columns
    leave them as they are
```

Your data is **not modified**. What you get is a recipe — a file you own:

```yaml
steps:
  # column was entirely missing
  - op: drop
    columns: ["notes"]
  # suspected target leakage
  - op: drop
    columns: ["outcome_code"]
  - op: impute
    columns: ["salary", "age"]
    strategy: median
```

The session ends by printing the command that reproduces it, so exploration
converts to a script:

```
  To reproduce this without the prompts:
    michi clean data/customers.csv --drop notes,outcome_code --impute salary=median …
```

## 3. Compare some models

```bash
michi bench data/customers.csv --target purchased \
  --recipe michi.recipe.yaml --models linear,rf,hist-gbm
```

```
  Results  (ranked by balanced_accuracy)

  model      balanced_accuracy      95% interval   vs leader                fit
  ────────────────────────────────────────────────────────────────────────────
  linear                 0.708   0.6278 – 0.7888   leader                  0.1s
  hist-gbm               0.696   0.6058 – 0.7853   tied with leader (p=1)  0.4s
  rf                     0.694   0.6157 – 0.7723   tied with leader (p=1)  0.9s
  dummy                    0.5         0.5 – 0.5   worse (p=0.0261)        0.0s

  Verdict  linear scores highest, but hist-gbm, rf are statistically
  indistinguishable from it at this sample size. Choosing between them on
  these numbers alone is not supported.
```

Most tools would have declared `linear` the winner. Differences are tested
with the corrected resampled *t*-test, because cross-validation folds share
training data and a naive test calls noise significant. A dummy baseline is
always added, so "is this any good?" has an answer.

Your recipe's preparation is fitted **inside each fold**, so the comparison
cannot leak.

## 4. Keep the result

Every run wrote a manifest to `runs/`. Render them:

```bash
michi report runs/ --out report.html --open   # a self-contained page
michi report runs/ --format markdown          # for a pull request
michi report runs/ --format latex             # for a paper
```

## 5. Leave with code

```bash
michi export michi.recipe.yaml -o pipeline.py
```

You now have a standalone module that imports pandas and scikit-learn — never
michi. `prepare()` for the deterministic steps, `build_pipeline()` for the
ones that learn from data, split precisely because they carry different
leakage risks, with the reason written into the file.

If you stop using michi tomorrow, you keep working code.

## Where next

| You want to | Read |
|---|---|
| Score a model you already trained | [`michi eval`](eval.md) |
| Run a grid of experiments | [`michi sweep`](sweep.md) |
| Explore with tab completion over your columns | [the console](console.md) |
| Browse runs in a browser | [`michi ui`](ui.md) |
| Gate a model in CI | [`michi eval --fail-under`](eval.md#in-ci) |
| Add your own models or loaders | [plugins](plugins.md) |
| Know what michi will never do | [the philosophy](philosophy.md) |

## Using it in CI

```bash
michi inspect data/train.csv --target label --fail-on high --quiet
michi eval model.pkl data/test.csv --target label --fail-under f1=0.85
```

Exit codes are meaningful, `--json` is on every verb, and nothing is
interactive unless you ask for it.
