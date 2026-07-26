<div align="center">

# 道

### michi

*A local-first ML workbench —*
*automate the implementation, never the judgement.*

<br>

[![ci](https://github.com/cattolatte/michi/actions/workflows/ci.yml/badge.svg)](https://github.com/cattolatte/michi/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/cattolatte/michi?display_name=tag&sort=semver&color=b23a30)](https://github.com/cattolatte/michi/releases)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

<br>

Michi is a toolbox of independent command-line tools that automate the
repetitive implementation work of machine learning — profiling datasets,
cleaning them, evaluating models, benchmarking, running experiment grids, and
reporting — while leaving every judgement call to you.

It works on the projects you already have: your CSV, your `model.pkl`, your
repo. There is no project template to adopt, no framework to import, no
account to create, and nothing ever leaves your machine.

<br>

> ## 名前 — on the name
>
> **道** (*michi*) means "path" or "road". Read as **-dō**, the same character
> closes the names of Japanese disciplines — 柔道 (*jūdō*, the yielding way),
> 書道 (*shodō*, the way of writing), 茶道 (*chadō*, the way of tea) — where it
> means not merely a road, but *a practice one walks for oneself*.
>
> That is the entire design brief. Michi clears the path. You walk it.

<br>

## 使い方 — getting started

```bash
pip install michi
michi inspect data.csv --target label --explain
```

Optional extras: `michi[bench]` (XGBoost, LightGBM, CatBoost), `michi[excel]`,
`michi[shap]`, `michi[ui]`.

<br>

## 道具 — the toolbox

Every verb stands alone. Use one, ignore the rest.

| Verb | What it does | Status |
|---|---|:---:|
| `michi inspect data.csv` | Profile a dataset: types, missing values, duplicates, skew, imbalance, correlations, outliers, leakage suspects — every finding explained | **v0.1** |
| `michi eval model.pkl data.csv --target y` | Rigorously evaluate an existing model: metrics with intervals, calibration, baselines, subgroup gaps | **v0.2** |
| `michi bench … --models rf,linear,xgb` | Train and compare models with honest CV, confidence intervals, significance tests | **v0.3** |
| `michi clean` · `apply` · `export` | Interactive cleaning that authors a reproducible recipe and exports readable pipeline code | **v0.4** |
| `michi` | Interactive console with context-aware completion; every session exports to a replayable script | **v0.5** |
| `michi sweep sweep.yaml` | Reproducible experiment grids: models × recipes × seeds, with resume | **v0.6** |
| `michi report runs/` | HTML · Markdown · LaTeX reports over recorded runs | **v0.3** |
| `michi ui` | Local, read-only viewer over your runs | **v0.7** |

<br>

## 見本 — a look

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
  salary         numeric         11.7%      106   mean 38,169 · range 30,137–46,303
  cabin          categorical     87.5%       15   top: C0 (1), C8 (1), C16 (1)
  notes          empty          100.0%        0
  fare           numeric             —       11   mean 259 · skew +6.16 · 3 outliers
  …

  Findings (15)

  high    notes                  every value is missing
  high    country                only one distinct value (JP)
  high    cabin                  87.5% missing (105 of 120)
  high    outcome_code           each of its 2 values maps to exactly one
                                 'purchased' class
  warn    purchased              smallest class 8.3% vs largest 91.7%
  warn    age, age_months        correlation +1.000
  warn    signup_date            values parse as dates but are stored as text
  …

  Run again with --explain for what each finding means and your options.
```

`--html` writes a [self-contained offline report](examples/profile.html)
(34 KB, no CDN, no JavaScript). `--json` writes a
[machine-readable profile](examples/profile.json) you can diff in CI, and
`--fail-on high` turns michi into a data-quality gate.

Then compare some models — and find out whether the difference is real:

```bash
michi bench data/customers.csv --target purchased --models linear,rf,hist-gbm
```

```
 道  michi bench  ·  5 models

  classification  ·  490 rows  ·  5-fold cross-validation  ·  target purchased
  preparation: numeric: impute median · categorical: impute most_frequent +
  onehot · standardise (scale-sensitive models only) — fitted inside each fold

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
with the corrected resampled *t*-test (Nadeau & Bengio, 2003), because
cross-validation folds share training data and a naive test calls noise
significant.

Or run `michi` with no arguments for the console, where tab completion knows
*your* column names:

```
michi › use data/customers.csv
loaded customers.csv — 13 columns
michi (customers.csv) › set target purchased
michi (customers.csv → purchased) › bench --models linear,rf
…
michi (customers.csv → purchased) › history --export session.sh
  wrote session.sh — a replayable script of one-shot michi commands
```

The console is a skin over the same commands — it adds no capability, and
every session exports back to plain one-shot invocations.

Full options: [`michi inspect`](docs/inspect.md) · [`michi eval`](docs/eval.md) ·
[`michi bench`](docs/bench.md) · [`michi report`](docs/report.md) ·
[`michi clean`](docs/clean.md) · [`michi sweep`](docs/sweep.md) ·
[`michi ui`](docs/ui.md) · [the console](docs/console.md).

<br>

## 心得 — principles

**道具、流れにあらず** · *A toolbox, not a workflow.*
Use one verb, ignore the rest, keep your own project structure.

**献立、勧めにあらず** · *Menus, not recommendations.*
Michi lists the options; you choose. Defaults exist for mechanics — folds,
seeds — never for judgement.

**作品、記憶にあらず** · *Artifacts, not sessions.*
Every decision becomes a durable, versionable file you own.

**厳密さは既定** · *Rigor by default.*
Baselines, confidence intervals, significance tests, leakage checks — opt-out,
not opt-in.

**手元にて完結** · *Entirely local.*
No server, no account, no telemetry, no network call. Ever.

The full reasoning, and the list of things michi will deliberately never do,
lives in [the philosophy](docs/philosophy.md); what ships when is in
[the roadmap](docs/roadmap.md).

<br>

## 開発 — development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run mypy src/michi
uv run pytest
```

All four gates run in CI on Linux, macOS, and Windows, against Python 3.11
and 3.13.

<br>

## ライセンス — license

[MIT](LICENSE)

<br>

<div align="center">

*用の美* — beauty through use.

</div>
