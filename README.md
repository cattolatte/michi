<div align="center">

# 道 michi

**A local-first ML workbench — automate the implementation, never the judgement.**

[![ci](https://github.com/cattolatte/michi/actions/workflows/ci.yml/badge.svg)](https://github.com/cattolatte/michi/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/cattolatte/michi?display_name=tag&sort=semver)](https://github.com/cattolatte/michi/releases)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

---

Michi is a toolbox of independent command-line tools that automate the
repetitive implementation work of machine learning — profiling datasets,
cleaning them, evaluating models, benchmarking, running experiment grids, and
reporting — while leaving every judgement call to you.

It works on the projects you already have: your CSV, your `model.pkl`, your
repo. There is no project template to adopt, no framework to import, no
account to create, and nothing leaves your machine.

> ### On the name
>
> **道** (*michi*) means "path" or "road". Read as **-dō**, the same character
> ends the names of Japanese disciplines — 柔道 (*jūdō*, the gentle way),
> 書道 (*shodō*, the way of writing), 茶道 (*chadō*, the way of tea) — where it
> means not merely a road but *a practice one walks for oneself*.
>
> That is the whole design brief. Michi clears the path; you walk it.

## Install

```bash
pip install michi
```

Optional extras: `michi[bench]` (XGBoost, LightGBM, CatBoost), `michi[excel]`,
`michi[shap]`, `michi[ui]`.

## The toolbox

| Verb | What it does | Status |
|---|---|---|
| `michi inspect data.csv` | Profile a dataset: types, missing values, duplicates, skew, imbalance, correlations, outliers, leakage suspects — every finding explained | ✅ **v0.1** |
| `michi eval model.pkl data.csv --target y` | Rigorously evaluate an existing model: metrics, calibration, baselines, leakage checks | 🔜 v0.2 |
| `michi bench … --models rf,logreg,xgb` | Train and compare models with honest CV, confidence intervals, significance tests | 🔜 v0.3 |
| `michi clean` / `apply` / `export` | Interactive cleaning that authors a reproducible recipe and exports readable pipeline code | 🔜 v0.4 |
| `michi` (console) | Interactive shell with context-aware tab completion; every session exports to a replayable script | 🔜 v0.5 |
| `michi sweep sweep.yaml` | Reproducible experiment grids: models × recipes × seeds | 🔜 v0.6 |
| `michi report runs/` | HTML / Markdown / LaTeX reports over recorded runs | 🔜 v0.3 |
| `michi ui` | Local read-only viewer over your runs | 🔜 v0.7 |

## Quick look

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

Add `--html profile.html` for a [self-contained offline report](examples/profile.html)
(34 KB, no CDN, no JavaScript), or `--json profile.json` for a
[machine-readable profile](examples/profile.json) you can diff in CI. In a
pipeline, `--fail-on high` turns michi into a data-quality gate.

Full options: [`michi inspect` documentation](docs/inspect.md).

## Philosophy

- **Toolbox, not workflow** — use one verb, ignore the rest, keep your project structure.
- **Menus, not recommendations** — michi lists the options; you pick. Defaults exist for mechanics (folds, seeds), never for judgement.
- **Artifacts over sessions** — every decision becomes a durable, versionable file.
- **Rigor by default** — baselines, confidence intervals, significance tests, leakage checks: opt-out, not opt-in.
- **Local-first** — no server, no accounts, no telemetry, no network calls. Ever.

Read [the philosophy](docs/philosophy.md) for the full reasoning and the list
of things michi will deliberately never do, and [the roadmap](docs/roadmap.md)
for what ships when.

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run mypy src/michi
uv run pytest
```

All four gates run in CI on Linux, macOS, and Windows.

## License

[MIT](LICENSE)
