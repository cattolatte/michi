# Michi (道)

> A local-first ML workbench: independent command-line tools that automate the
> repetitive implementation work of machine learning — inspecting data,
> cleaning it, evaluating models, benchmarking, running experiment grids, and
> reporting — while leaving every judgement call to you.
>
> **Automate implementation. Never automate judgement.**

Michi (道, "the path") makes the ML journey easier without ever walking it for
you. It is a **toolbox, not a workflow**: every verb works standalone, on
artifacts you already have — your CSV, your `model.pkl`, your existing repo.
No project template, no accounts, no cloud, no telemetry. Everything is local,
reproducible, and explained.

## Status

**Pre-alpha.** The project is in its skeleton phase; `v0.1` (`michi inspect`)
is the first milestone. See [PLAN.md](PLAN.md) — the master planning document —
for the full vision, architecture, and roadmap.

## The toolbox (planned)

| Verb | What it does | Milestone |
|---|---|---|
| `michi inspect data.csv` | Profile a dataset: types, missing values, skew, imbalance, correlations, outliers — every finding explained | v0.1 |
| `michi eval model.pkl data.csv --target y` | Rigorously evaluate an existing model: metrics, calibration, baselines, leakage checks | v0.2 |
| `michi bench … --models rf,logreg,xgb` | Train and compare models with honest CV, confidence intervals, and significance tests | v0.3 |
| `michi clean` / `apply` / `export` | Interactive cleaning that authors a reproducible recipe, applies it non-destructively, and exports readable pipeline code | v0.4 |
| `michi` (console) | Interactive shell with context-aware tab completion; every session exports to a replayable script | v0.5 |
| `michi sweep sweep.yaml` | Reproducible experiment grids: models × recipes × seeds | v0.6 |
| `michi report runs/` | HTML / Markdown / LaTeX reports over recorded runs | v0.3+ |
| `michi ui` | Local read-only viewer over your runs | v0.7 |

## Philosophy

- **Toolbox, not workflow** — use one verb, ignore the rest, keep your project structure.
- **Artifacts over sessions** — every interactive decision becomes a durable, versionable file (recipe, manifest, report).
- **Rigor by default** — baselines, confidence intervals, significance tests, leakage checks: opt-out, not opt-in.
- **Local-first, offline, private** — no server, no accounts, no telemetry, ever.
- **Transparent and educational** — findings come with explanations; generated code is readable and yours.

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev      # install with dev tooling
uv run ruff check .      # lint
uv run ruff format --check .
uv run mypy src/michi    # strict type checking
uv run pytest            # tests (fully offline)
```

All four gates must pass before a change is considered done; CI enforces them
on Linux, macOS, and Windows.

## License

[MIT](LICENSE)
