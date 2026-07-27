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

## Artifacts

Everything michi produces is one of four durable, inspectable files:

- **Profile** (JSON) — what `inspect` learned about a dataset.
- **Recipe** (commented YAML) — the cleaning and preparation decisions you made.
- **Run manifest** (JSON) — one evaluation, benchmark cell, or sweep cell, with
  the hashes, seeds, and environment needed to reproduce it.
- **Report** (HTML / Markdown / LaTeX) — rendered from the above.

Schemas carry an explicit `schema_version` from v0.1 and are frozen under
semantic versioning at v1.0.
