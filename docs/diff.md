# `michi diff`

Has this data changed since the profile you trusted?

```bash
michi diff baseline.profile.json production.csv --target churned
michi diff last_month.csv this_month.csv
michi diff baseline.json today.csv --fail-on high    # a nightly CI gate
```

## Why it lives here

`inspect` has written schemas, distributions, and hashes since v0.1. Comparing
two of those profiles needs **no new measurement** — which is exactly why a
drift check belongs in michi rather than in a monitoring service michi would
have to run, and would then be in the business of operating.

Either side may be a committed `profile.json` or a raw data file. A profile is
the better baseline: it is small, diffable, and already in your repository.

## What it reports

```
 道  michi diff

  base.json → drifted.csv   ·   600 → 600 rows

  high   cabin    1 column(s) present in the baseline are gone: cabin
  warn   salary   salary mean moved +5.35 baseline SD (3.936e+04 → 7.478e+04)
  warn   region   region has value(s) absent from the baseline: NORTH_NEW
  info   extra    1 new column(s): extra
```

| Finding | Severity | Why |
|---|---|---|
| `column-removed` | high | Breaks the model *today*, whatever the distributions did |
| `type-changed` | high | Two kinds are not comparable; the numeric checks are skipped |
| `missingness-changed` | warn | A column that started arriving null |
| `distribution-shifted` | warn / info | Mean moved, measured in **baseline standard deviations** |
| `new-categories` | warn | An encoder has no slot for a level it never saw |
| `cardinality-changed` | warn | The number of distinct values moved by half |
| `category-share-changed` | info | The majority level changed hands |
| `row-count-changed` | warn | Halved or doubled — usually a broken export, not drift |
| `column-added` | info | A fitted model never asked for it, so it cannot break |

Three decisions shape that table.

**Severity follows breakage, not effect size.** A removed column outranks a
large distribution shift, because one stops the model running today and the
other might never matter.

**Shift is measured in baseline standard deviations**, so a single threshold
means the same thing for a column in dollars and one in millimetres.

**Incomparable columns are not compared.** If a column was numeric and is now
text, michi says so and stops — the distance between a string and a float is
not a number anyone should act on.

## As a CI gate

```bash
michi diff data/baseline.profile.json data/latest.csv --fail-on high
```

Exit code 1 when anything at or above that severity moved, 0 otherwise. The
command a person runs by hand is the one CI runs nightly.

## What it will not do

It reports; it does not repair. Whether a shifted column matters depends on
what it feeds and what the model is for, and michi knows neither.

## Options

| Option | Default | Purpose |
|---|---|---|
| `--target`, `-t` | none | Label column, when profiling raw data |
| `--json` | none | Write the comparison as JSON |
| `--fail-on` | none | Exit non-zero at `high` or `warn` |
| `--sample` / `--full` | — | Sampling for large files |
