# Roadmap

Each milestone ships one independently useful verb as a complete vertical
slice — command, logic, artifact, documentation, tests — and leaves michi
releasable. Releases happen when a milestone's bar is met, never by date.

**Every milestone below has shipped.** From 1.0 the artifact schemas, the CLI
surface, and the plugin contract are frozen under semantic versioning
([ADR-0002](adr/0002-freeze-the-public-surface.md)); growth now happens at the
edges, in plugins, rather than in the core.

| Version | Verb | What it delivers |
|---|---|---|
| **v0.1** | `inspect` | Dataset profiling: types, missing values, duplicates, cardinality, skew, imbalance, correlations, outliers, leakage suspects. Terminal, HTML, and JSON output with explanations attached to findings. |
| **v0.2** | `eval` | Rigorous evaluation of an existing model: task-appropriate metrics, calibration, per-slice performance, an always-included dummy baseline, and run manifests. |
| **v0.3** | `bench` + `report` | Training and honest comparison of several models with proper cross-validation, confidence intervals, and significance tests; HTML, Markdown, and LaTeX reports. |
| **v0.4** | `clean` / `apply` / `export` | Findings-driven interactive cleaning that authors a reproducible recipe, applies it non-destructively, and exports readable pipeline code. |
| **v0.5** | console | Interactive shell with context-aware tab completion; every session exports to a replayable script. |
| **v0.6** | `sweep` | Declarative experiment grids — models × recipes × seeds — with caching and resume. |
| **v0.7** | `ui` | Local, read-only viewer over recorded runs. |
| **v0.8** | plugins | Entry-point discovery for models and model loaders, with a published compatibility suite plugin authors run themselves. |
| **v1.0** | freeze | Artifact schemas, the CLI surface, and the plugin contract come under semantic versioning. **Shipped.** |
| **v1.1** | feature engineering + `path` | `datepart`, `log`, `interact`, `binarize`, `bin` as recipe operations; `path` and `walk` in the console. **Shipped.** |
| **v1.2** | teaching mode | `bench --explain` explains the run's own numbers: the gap over the baseline, the fold spread behind the interval, why two models tied, and the rows it would take to separate them. **Shipped.** |

## Known gaps

Stated plainly, because a roadmap that only lists wins is marketing.

**Target encoding.** The highest-value categorical encoding for tabular ML,
and the one most likely to leak. scikit-learn deprecated the parameter that
makes `TargetEncoder` reproducible in 1.9 and removes it in 1.11; the
replacement needs a newer scikit-learn than michi's `>=1.4` floor. The
implementation is straightforward — the fitted/deterministic split already
puts it in the right place — but a step whose output changes between runs
would break the reproducibility everything else rests on. It ships when the
floor moves.

**Hyperparameter search is grid-only.** `sweep` enumerates models × recipes ×
seeds. Bayesian or successive-halving search would be a real addition, and
would need an ADR: it introduces an optimiser with an opinion, which is close
to the line michi draws.

**Deliberately absent, not missing:** deep-learning training loops, model
serving, monitoring, cloud anything. These are non-goals under PLAN §3 and
§11, not work that has not happened yet.

## Artifacts

Everything michi produces is one of four durable, inspectable files:

- **Profile** (JSON) — what `inspect` learned about a dataset.
- **Recipe** (commented YAML) — the cleaning and preparation decisions you made.
- **Run manifest** (JSON) — one evaluation, benchmark cell, or sweep cell, with
  the hashes, seeds, and environment needed to reproduce it.
- **Report** (HTML / Markdown / LaTeX) — rendered from the above.

Schemas carry an explicit `schema_version` from v0.1 and are frozen under
semantic versioning at v1.0.
