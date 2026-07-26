# Michi (道) — Project Plan

> **Michi** (道, "the path") is a local-first ML workbench: a set of independent
> command-line tools that automate the repetitive implementation work of
> machine learning — inspecting data, cleaning it, evaluating models,
> benchmarking, running experiment grids, and reporting — while leaving every
> judgement call to the human.
>
> Michi makes the path easier. It never walks it for you.
>
> **Automate implementation. Never automate judgement.**

This is the master planning document. It records what we are building, why, in
what order, and — just as importantly — what we refuse to build. Sections
marked **LOCKED** are settled; reopening one requires an ADR in `docs/adr/`
explaining what changed since the decision was made.

---

## 1. Vision

Everyone doing ML — students, Kagglers, researchers, production engineers —
performs the same tasks on every project: profile the data, handle missing
values, encode, split, train candidate models, compare them honestly, make the
plots, write up the results. None of it is intellectually hard. All of it is
time-consuming, and most of it is re-implemented from memory, slightly wrong,
every single time.

Michi is a **toolbox, not a workflow**. Each tool is a verb that does one job
well, works on artifacts the user *already has* (their CSV, their `model.pkl`,
their scripts, their existing repo), and produces durable, inspectable outputs.
Use one verb and ignore the rest forever. Nothing is hidden: every automated
step is explained, every interactive decision is captured as a re-runnable
artifact, and every result is reproducible from files on disk.

Five-year picture of success: Michi is boring, trusted infrastructure — the
thing you reflexively run before starting EDA (`michi inspect`), before
claiming one model beats another (`michi bench`), and before writing a paper's
results section (`michi report --format latex`). Adopted one verb at a time,
never as a leap of faith.

## 2. Mission

Give every ML practitioner back the hours currently spent on repetitive
implementation — without taking away a single decision that is theirs to make.

## 3. Philosophy (LOCKED)

1. **Automate implementation, never judgement.** Tools observe, report, and
   offer options; the user decides; the tool executes and records.
2. **Toolbox, not workflow.** Every verb is independently useful. No forced
   order, no required project structure, no init step.
3. **Artifacts over sessions.** Every interactive flow compiles to a durable,
   versionable artifact (recipe, manifest, report, generated code). The
   session is the authoring UI; the artifact is the product.
4. **Transparency beats convenience** whenever they conflict.
5. **Rigor by default.** Baselines, confidence intervals, significance tests,
   leakage checks — opt-out, not opt-in.
6. **Local-first and offline.** No server required, no accounts, no telemetry,
   no network calls. State is files in the user's repo; git is the history.
7. **Education through observation.** Explanations attach to findings, never
   to recommendations Michi didn't make.

## 4. Design principles (engineering)

- **Concrete before abstract.** One real implementation before any interface.
- **Evidence-driven abstraction.** A protocol/base class requires two concrete
  implementations that demand it.
- **No speculative infrastructure.** Config systems, plugins, registries are
  built in the milestone where a real consumer first needs them.
- **Vertical slices.** Every milestone ships a complete, usable verb: CLI →
  logic → artifact → docs → tests. Michi is releasable at every milestone.
- **Own the interface.** pandas/sklearn/plotly are implementation details;
  module boundaries expose Michi-native types; optional deps import lazily;
  third-party failures are wrapped in `MichiError` subclasses.
- **Readability over cleverness.** Michi's outputs (and code) are teaching
  texts that happen to run.
- **No hidden state.** Anything Michi "remembers" is a plain file the user can
  read, edit, diff, and delete.
- **Cross-platform from day one.** macOS, Linux, and Windows are all
  first-class: pure-Python core, `pathlib` everywhere, no shell-outs, explicit
  UTF-8 encodings, and all three OSes in the CI matrix from v0.1 — Windows is
  never a fix-on-complaint platform.

## 5. Target audience

| User | What they'll actually use | Design consequence |
|---|---|---|
| Repeat practitioners (data scientists, analysts, consultants, Kagglers) | everything; `bench` + `report` daily | **Optimize for these first** — they drive adoption and recommendations |
| Students / learners | `inspect`, `clean`, `bench`, explanations | Explanations are additive, never blocking |
| Researchers | `sweep`, manifests, `report --format latex` | Reproducibility must be bit-for-bit; LaTeX/markdown export first-class |
| Experienced ML engineers | `eval`, leakage checks, `--json` in CI | No capability may be interactive-only |
| Automation and coding agents | every verb via flags + `--json` | Machine-readable output and exit codes are a primary interface, not an afterthought |

## 6. Use cases (the adoption stories)

1. *"I just got a CSV."* → `michi inspect data.csv` — ten seconds to a profile
   with findings explained. First contact; zero decisions required.
2. *"I have a model and a test set — is it any good?"* → `michi eval model.pkl
   data.csv --target y` — honest metrics, baseline comparison, calibration,
   slices, leakage checks. Works on a model Michi never saw trained.
3. *"Which of these five models should I use?"* → `michi bench … --models
   rf,logreg,xgb --cv 5` — proper CV, significance tests, one report.
4. *"Clean this the same way every time."* → `michi clean` (interactive once)
   → `recipe.yaml` → `michi apply` forever after, in scripts and CI.
5. *"Run the full grid for the paper."* → `michi sweep sweep.yaml` →
   reproducible manifests → `michi report --format latex`.
6. *"Gate the model in CI."* → `michi eval … --fail-under f1=0.85 --json`.

## 7. Non-goals (LOCKED)

Each converts Michi into a product that already exists or fails structurally.

- **No AutoML.** Michi never searches model space uninvited or picks winners.
  It executes the comparisons *you* specify. (AutoGluon/FLAML exist; automating
  judgement violates the philosophy.)
- **No workflow ownership.** No required project template, no DAG engine, no
  orchestrator, no `michi run`-owns-your-main. (Kedro's ceiling.)
- **No training-loop wrapper.** Never `import michi` around your training
  code. (PyCaret's trap: leaky abstraction chained to sklearn's releases.)
- **No cloud, accounts, hosted service, or telemetry.** Ever.
- **No production serving or monitoring.** Evaluating a model on a dataset is
  in scope; watching live traffic is Evidently/Grafana's product.
- **No notebook environment or IDE.**
- **No hidden session state** (see §12).
- **No feature whose pitch begins "Michi manages X for you"** where X is the
  user's code, data decisions, or modeling judgement.

## 8. The artifact model (architectural center)

Four artifact types; every verb reads and/or writes them.

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| **Profile** | JSON (+ terminal/HTML rendering) | `inspect` | `report`, `clean`, humans |
| **Recipe** | Commented YAML — ordered declarative cleaning/prep operations + a schema snapshot of the data it was authored against | `clean` | `apply`, `bench`, `sweep`; compiles to a readable sklearn `Pipeline` script via `michi export` |
| **Run manifest** | JSON — dataset hash, recipe hash, model spec, seed, environment, git SHA, metrics, timings | `eval`, `bench`, `sweep` | `report`, `runs`, `ui` |
| **Report** | Self-contained HTML / Markdown / LaTeX tables | `report` (auto-emitted by `bench`) | humans |

Consequences:

- Reproducibility is structural: any run re-executes from manifest + recipe + data.
- `clean` is non-destructive by construction — it authors a recipe; only
  `apply` produces data, always to a new file.
- The recipe's schema snapshot doubles as a data contract: `apply` validates
  incoming data against it, so cleaning decisions survive to inference time.
- The dashboard (later) is a stateless viewer over these files.
- All schemas carry `schema_version` from day one; frozen under semver at v1.0.

## 9. CLI philosophy & surface

```
michi                                       # no args → interactive console (see below)
michi inspect data.csv                      # profile + explained findings; --html --json
michi clean data.csv                        # interactive triage → michi.recipe.yaml
michi clean data.csv --drop id --impute age=median --encode country=onehot
michi apply recipe.yaml data.csv -o clean.parquet
michi export recipe.yaml -o pipeline.py     # recipe → commented sklearn Pipeline code
michi eval model.pkl data.csv --target y    # rigorous single-model evaluation
michi bench data.csv --target y --models rf,logreg,xgb --cv 5
michi sweep sweep.yaml                      # models × recipes × seeds grid
michi report runs/ [--format html|md|latex] [--open]
michi runs list | michi runs diff RUN1 RUN2
michi ui                                    # (later) local read-only viewer over runs/
```

Rules (LOCKED):

- **Flag parity.** Every interactive flow has a complete non-interactive
  equivalent, and every interactive session *prints its own equivalent
  command* on exit ("to reproduce: `michi clean data.csv --impute age=median …`").
- `--json` on every verb; meaningful exit codes; `--seed` and `--no-input`
  global; `--fail-under metric=value` for CI gates.
- No destructive operations: outputs go to new files unless `-o` explicitly
  overwrites.
- Interactive cleaning is **findings-driven triage, not a column interrogation**:
  the wizard walks the *issues* `inspect` found (grouped, with bulk actions —
  "these 6 columns have <1% missing: impute median for all? [y/n/each]"),
  never all N columns one by one. A 200-column dataset must be cleanable in
  minutes.
- **Menus, not recommendations.** Wherever a judgement call exists, Michi
  lists the options with one-line factual descriptions and the user picks.
  Menus are *contextual*, never encyclopedic: imputation options appear only
  for columns that actually have missing values, imbalance options only when
  imbalance was detected. Every menu is discoverable non-interactively
  (`michi bench --list-models`, `michi clean --list-ops`). Defaults exist for
  mechanics (CV folds, seeds) — never for judgement.
- Large-data ergonomics: `inspect` samples beyond a size threshold (clearly
  labeled, `--full` to override) so first contact is always fast.

### The console (`michi` with no arguments)

An interactive shell in the spirit of `msfconsole` — the exploratory home for
a working session, with the same relationship to the one-shot CLI that
Metasploit's console has to its modules: a *skin*, never a second brain.

```
        道  michi v0.5.0

michi › use data/train.csv
[*] loaded train.csv — 14,204 rows × 27 cols (3 findings; run `inspect`)
michi (train.csv) › set target purchased
michi (train.csv → purchased) › inspect
…
michi (train.csv → purchased) › bench --models rf,logreg,xgb
…
michi (train.csv → purchased) › history --export session.sh
[*] wrote session.sh — replayable one-shot CLI script
```

Design rules (LOCKED):

- **Zero logic in the console.** It is a dispatcher + completer + context
  holder over the *same functions* the one-shot CLI calls. Any capability
  must exist as a flag before it may appear in the console.
- **Context is visible in the prompt** (`michi (train.csv → purchased) ›`) and
  dumpable via `show context` — Metasploit's `show options` lesson, improved:
  state is never hidden even one keystroke away.
- **Console state IS the `michi.toml` model**, held in memory: `set target y`
  edits it, `save` writes it to disk, exit offers to save. One state model
  everywhere; nothing dies silently with the session.
- **Every session is exportable** as a replayable shell script of equivalent
  one-shot commands (`history --export`) — the resource-script lesson.
  Exploration stays reproducible.
- **Tab completion is the killer feature** and completes *the user's actual
  world*: their column names once a dataset is loaded, model names, recipe
  ops, file paths. This is the one thing a one-shot CLI can never offer.
- Banner art: tasteful, one screen max, suppressed by `--quiet` and in
  non-TTY contexts.

What we copy from Metasploit: the visible-state grammar (`use`/`set`/`show`),
resource scripts, completion, a consistent verb language. What we reject:
interactive-only capabilities and session state that isn't captured anywhere
durable — the two things that would quietly turn Michi into a workflow owner.

## 10. Dashboard philosophy

**Static first, server last, deletable always.**

- Phases 1–2: reports are **self-contained HTML files** (plotly embedded,
  offline-safe). `--open` launches the browser. No server exists.
- v0.7: `michi ui` — a local, read-only viewer over `runs/` for interactive
  experiment comparison. FastAPI + server-rendered Jinja + htmx: maintainable
  forever by Python contributors, no node toolchain, no build step, no
  database (the runs directory is the store).
- **Deletability bar (LOCKED):** removing `ui` from the codebase must remove
  zero capabilities, only convenience — everything it shows exists as files.
  The moment the UI holds state or exclusive features, it has become a
  platform and violates §7.

## 11. Model & data support policy (LOCKED for v1)

**Models:** (a) sklearn-compatible estimators via pickle/joblib;
(b) *anything else* via a Python protocol — `--model mymodule:obj` where the
object exposes `predict(X)` (optionally `predict_proba`). That covers PyTorch,
TensorFlow, ONNX, and custom models *today* with zero per-framework loaders.

The "popular models" menu (`--list-models`) is a **built-in algorithm
registry** — Random Forest, Logistic Regression, Gradient Boosting in core;
XGBoost/LightGBM/CatBoost via `[bench]` — these are algorithms that train
locally on the user's data, so **nothing is ever downloaded from the
internet**. Pretrained/foundation-model adapters (TabPFN, hub models) are a
V2 idea (§23): any such download must be explicit, cached, and never required,
preserving offline-first.
Native ONNX may come later behind an extra and an ADR. "Load any `.pt`/SavedModel
file" is **never** promised — those formats aren't self-describing; pretending
otherwise is an unbounded support tarpit.

**Data:** CSV and Parquet in core (pandas + pyarrow). Excel behind a tiny
extra (`michi[excel]`). SQL **postponed**: connection strings drag in drivers,
auth, and dialects; the workaround (`export query → parquet`) is one line.
Revisit with evidence.

**Heavy deps are extras with graceful degradation:** `[bench]` → xgboost,
lightgbm, catboost; `[explain-shap]` → shap; `[ui]` → fastapi, uvicorn;
`[excel]` → openpyxl. A missing extra never breaks a command that doesn't
need it; the error message names the exact `pip install` to run.

## 12. Project defaults & session management

Evaluated and decided: **no hidden session state — but yes to visible defaults.**

The UX pain a "session" solves is real (retyping `--target y --recipe r.yaml`
on every command). The classic solution — a stateful session Michi remembers
(PyCaret's `setup()` global, a `.michi/session` blob) — is rejected
permanently: hidden state kills reproducibility ("works in my session"),
breaks scripting and CI, confuses agents, and rots.

Instead: an **optional, human-readable `michi.toml`** in the working directory
supplying *defaults for CLI flags only*:

```toml
[defaults]
data = "data/train.parquet"
target = "purchased"
recipe = "michi.recipe.yaml"
runs_dir = "runs/"
```

Plain file, created by the user or offered by a wizard on exit ("save these as
project defaults? [y/N]"), checked into git, diffable in PRs, always
overridden by explicit flags, never required. Precedence: flags > `michi.toml`
> built-ins — printed by `michi info` so there is never a mystery about where
a value came from.

The console (§9) holds exactly this model in memory: `set`/`show context`
operate on it, `save` persists it. One state model across both interfaces.
"Session management" beyond this is a **non-goal** (§7).

## 13. Repository & module architecture

```
michi/
├── src/michi/
│   ├── core/            # artifact dataclasses, hashing, io, errors, versioned schemas, michi.toml resolution
│   ├── inspection/      # profiling: stats, correlations, outliers, imbalance, sampling
│   ├── recipes/         # recipe model, ops (impute/encode/scale/drop/…), apply, export-to-code
│   ├── evaluation/      # model evaluation: metrics, baselines, significance, leakage checks
│   ├── bench/           # multi-model training/comparison, built on evaluation's public surface
│   ├── sweep/           # grid execution over recipes × models × seeds → manifests, caching, resume
│   ├── report/          # renderers: terminal (rich), HTML (jinja+plotly), markdown, latex
│   ├── explain/         # explanation content (data files) + attachment logic
│   ├── adapters/        # model loading: sklearn-pickle, module:obj protocol, (later) onnx
│   └── cli/             # typer app; thin — parse args, call modules, render
│
│   # Package names avoid shadowing the stdlib/builtins (`inspect`, `eval`);
│   # the CLI verbs remain `michi inspect` / `michi eval`.
├── tests/{unit,golden,e2e}/
├── docs/{adr,…}         # ADRs from day one; mkdocs site
├── examples/            # committed example outputs: recipes, manifests, reports
└── PLAN.md
```

- **Dependency direction (one-way):** `core` → domain modules → `cli`. Domain
  modules never import each other's internals; they communicate **through
  artifacts**, not shared objects. (`bench` uses `eval`'s public API; `sweep`
  uses `bench`'s.) There is deliberately no cross-cutting "MichiContext" god
  object — that's how frameworks are born.
- **Configuration:** Michi itself needs almost none (`michi.toml` defaults,
  §12). The *user's* configuration lives in their recipes and sweep files,
  where they can read it.
- **Logging:** the CLI prints what it does (rich); runs log to their manifest.
  No logging framework, no log files, no verbosity matrix beyond `-q`/`-v`.
- **Error handling:** `MichiError` hierarchy at public boundaries; third-party
  failures wrapped and chained (`raise … from e`); messages are actionable.
- **Plugin strategy:** dormant until v0.8 (evidence-driven — the internal
  interfaces must survive six milestones of real use first). Then Python
  entry points: `michi.adapters`, `michi.recipe_ops`, `michi.report_sections`,
  plus a published contract-test suite (`pytest --michi-compat`) so plugin
  authors CI themselves and the maintainer is not the integration point.
- **State management:** there is none beyond files (artifacts, `michi.toml`,
  `runs/`). This sentence is the design.

## 14. Technology stack

| Decision | Choice | Why (and what was rejected) |
|---|---|---|
| Python floor | ≥ 3.11 | `Self`, exception groups, perf; a 3.12-only floor costs real adoption in university/cluster environments for little gain. |
| Packaging / dev | `uv`, hatchling | Fast, lockfile-driven, the current default. |
| Lint / format / types | `ruff` (lint **and** format), `mypy --strict` | One tool for style (black rejected: redundant beside ruff-format); strict typing from day one is cheap now, impossible to retrofit. |
| CLI | Typer + Rich | Type-hint-driven commands match a strictly-typed codebase; Rich renders tables/progress. (argparse: ceremony; click: Typer wraps it.) |
| Interactive prompts | `questionary` | Checkbox/select UIs Rich lacks; tiny; used only inside `clean`. (Textual TUI rejected: an app framework for what is one wizard.) |
| Console (v0.5) | `prompt_toolkit` + Rich rendering | The standard for Python REPLs (questionary already depends on it, so it's effectively free); gives completion, history, and bottom-toolbar control. (cmd/cmd2 rejected: weaker completion, dated UX.) |
| DataFrames | pandas + pyarrow engine | sklearn interop *is the product*; polars would add a translation layer users pay for. Revisit only via ADR with evidence. |
| Artifact models | `@dataclass(frozen=True, slots=True)` + explicit validation + `schema_version` | Keeps core dependency-light and validation readable/teachable. (pydantic rejected for core: heavy dep to avoid writing `__post_init__`; public schemas shouldn't be coupled to a third-party model class.) |
| Recipes | YAML written through templates (so emitted recipes carry explanatory comments) | Comments are the teaching surface; `yaml.dump` can't produce them. |
| Plots / reports | plotly with embedded JS (self-contained files); `--light` for matplotlib-SVG | Offline requirement forbids CDN loads; a ~4 MB file that always opens beats a small one that needs network. |
| Templating | Jinja2 | Boring and universal. |
| Dashboard (v0.6) | FastAPI + Jinja + htmx | Server-rendered HTML a Python community can maintain for years; SPA + node toolchain rejected as a rot vector. |
| Docs | mkdocs-material, Diátaxis | Standard, low-friction. |
| Testing | pytest + syrupy (snapshots) + hypothesis (property tests) | See §17. |

## 15. Feature roadmap & milestones

Each milestone ships one independently useful verb as a vertical slice.
Release when the bar is met — never date-driven.

### v0.1 — `inspect` (the wedge)
Profile any CSV/Parquet: dtypes, missing, duplicates, cardinality, skew,
imbalance, correlations, constant columns, outliers, datetime detection;
sampling for large files. Terminal + `--html` + `--json`. Explanations
attached to findings. `core/` artifacts + hashing land here.
**Bar:** runs on 10 messy real Kaggle datasets without crashing; the output
teaches a student something and doesn't insult an expert; golden tests green;
quickstart live.

### v0.2 — `eval` + run manifests
Evaluate an existing model (pickle or `module:obj`): task-appropriate metrics,
calibration, confusion, per-slice metrics, **dummy baseline always included**,
leakage checks. Manifests to `runs/`; `--fail-under` CI gates. Manifest schema
draft published.
**Bar:** an experienced DS learns one thing they didn't know about their model.

### v0.3 — `bench` + `report`  *(end of MVP)*
Train + compare user-specified models (sklearn core; `[bench]` extra) with
proper CV, paired significance tests (fold-correlation-corrected), confidence
intervals, plain-language honesty ("B beats A, but not significantly").
`report` renders manifests → HTML/markdown; LaTeX tables.
**Bar:** the report alone convinces a skeptic; a researcher pastes the LaTeX
table into a paper unedited.

### v0.4 — `clean` / `apply` / `export` (recipes)
Findings-driven interactive triage → commented `recipe.yaml`; full flag
parity; non-destructive `apply` with schema validation; `export` compiles the
recipe to a readable sklearn `Pipeline` script. `bench`/`eval` accept
`--recipe`. `michi.toml` defaults ship here.
**Bar:** wizard prints its own non-interactive equivalent; recipes round-trip
(author → apply ≡ export → run); mutation tests pass on hand-edited recipes.

### v0.5 — the console
`michi` with no args → interactive shell (§9): prompt-visible context,
`use`/`set`/`show context`, tab completion over the user's actual columns,
models, and ops; `history --export` to a replayable script; state persisted
via `michi.toml`. Zero logic — a skin over the existing verbs.
**Bar:** every console session is exportable to a one-shot script that
reproduces it exactly; deleting the console removes no capability.

### v0.6 — `sweep` (research mode)
Declarative `sweep.yaml`: models × recipes × seeds. Sequential/joblib-parallel,
content-hash caching, resume after interrupt, manifests per cell.
**Bar:** a 30-cell sweep reproduces bit-for-bit from yaml + data.

### v0.7 — `ui` (local viewer)
Read-only FastAPI+htmx viewer over `runs/`: comparison, curves, importance,
SHAP via extra. **Bar:** the deletability test (§10) holds.

### v0.8 — plugin surface
Entry-point discovery + published compat suite. First community surface;
contributor and plugin-author docs written now, not before.

### v1.0 — freeze
Manifest/recipe/profile schemas + CLI surface under semver. Ships only after
external users have shaped 0.4–0.8.

## 16. MVP definition

**MVP = v0.1–v0.3: `inspect`, `eval`, `bench`+`report`.**

Deliberately *not* the cleaning wizard: inspection and honest evaluation
deliver value in the first ten minutes on an existing project with **zero
decisions required from the user** — the fastest possible trust loop. Cleaning
asks users to hand Michi their data decisions; that trust must be earned first.

Explicitly not in MVP: recipes, the console, sweep, dashboard, plugins, ONNX,
SHAP, HPO, Excel/SQL, any deep-learning-specific support, `michi.toml`.

## 17. Testing strategy

- **All tests offline.** Tiny fixture datasets committed in-repo, including
  pathological ones (mixed types, unicode, single row, all-null columns,
  planted target leakage).
- **Unit:** public contracts only; behavioral assertions (shapes, invariants,
  "beats the dummy baseline"), never exact floats. Function-based, one
  behavior per test, docstring per test.
- **Golden/snapshot (syrupy):** terminal output, HTML reports (timestamps
  masked), emitted recipes, generated pipeline code — the main regression net
  for a tool whose product is its output.
- **Property-based (hypothesis):** recipe ops — deterministic, never touch the
  target, never read test-fold data; `apply(export(recipe))` ≡ `apply(recipe)`.
- **E2E:** real CLI on fixtures (inspect → eval → bench → report), then
  *execute the exported pipeline script* and assert it matches `apply`.
- **Generated-output quality gate:** exported code passes ruff + mypy in CI.
- **Statistics validation:** significance machinery tested against published
  known cases — wrong rigor is worse than none.
- **Compatibility matrix:** Linux, macOS, and Windows runners × oldest-supported
  and latest pandas/sklearn; extras in separate CI jobs. Console and wizard
  smoke-tested on Windows terminals specifically.
- **Plugin compat suite** from v0.8, run by plugin authors in their own CI.

## 18. Documentation strategy

- Diátaxis on mkdocs-material: **Quickstart** (one messy CSV → inspect → bench
  → report in 15 minutes; this page decides adoption — over-invest),
  **How-to** per verb, **Reference** (CLI + artifact schemas),
  **Explanations** that teach the ML itself (why stratify, reading calibration,
  what leakage looks like). The `explain/` content links into these pages —
  the education mission lives in both.
- `examples/` holds committed real outputs so the product is readable on
  GitHub before installing.
- Every module: README (purpose + public surface + what it deliberately does
  not do). Every major decision: ADR. Docs land in the same PR as the feature.
- Contributor/plugin docs at v0.8 (docs written before their audience exists rot).

## 19. Coding standards

- Python ≥ 3.11, `from __future__ import annotations`, full type hints,
  mypy strict, ruff lint+format, line length 88.
- Immutable value objects: `@dataclass(frozen=True, slots=True)`, invariants
  in `__post_init__`.
- `MichiError` hierarchy at public boundaries; wrap + chain third-party errors;
  actionable messages.
- NumPy-style docstrings; module docstrings include a *Design Principles*
  section stating what the module deliberately does **not** do.
- Lazy-import optional deps; Michi-native types at module boundaries.
- **Explanation content is data, not code** — structured files under
  `explain/content/`, reviewed like docs, so accuracy fixes never touch logic.

## 20. Release strategy

- Semver. Pre-1.0: minor = new verb, patch = fixes; breaking changes allowed
  with CHANGELOG migration notes. Post-1.0 frozen surface: CLI flags, artifact
  schemas, plugin contract.
- Every release: CI green (lint, types, tests, matrix), CHANGELOG, docs in the
  same PR, `uv build` + trusted-publisher PyPI upload from CI, git tag.
- Cadence: when the milestone bar is met — never by date.

## 21. Contributor guidelines (full CONTRIBUTING.md at v0.4)

- Read this file and `docs/adr/` first. PRs violating §3/§7 are declined with
  a citation, not a debate — "no" is written down so it's never personal.
- One verb-slice or one fix per PR; tests and docs land with the feature.
- New dependencies justified in the PR; new *core* dependencies need an ADR.
- Abstractions require two concrete uses, cited in the PR.
- Gates before review: `uv run ruff check`, `uv run ruff format --check`,
  `uv run mypy src/michi`, `uv run pytest`.

## 22. Risks (ranked by cost-to-fix-later)

1. **Framework accretion.** Every release someone proposes "Michi could also
   run/serve/orchestrate…". Mitigation: §7 is LOCKED and citable.
2. **Model-format tarpit.** Per-framework loaders each look small and each is
   unbounded. Mitigation: the `module:obj` protocol is the escape hatch;
   native loaders need an ADR + extra.
3. **Dashboard treadmill.** A live UI absorbs infinite effort. Mitigation:
   static-first; the deletability bar (§10).
4. **Schema freeze timing.** Too early → mistakes forever; too late → nothing
   builds on them. Mitigation: versioned from v0.1, draft at v0.2, frozen at
   v1.0 after external use.
5. **Rigor theater.** Wrong statistics are worse than none. Mitigation:
   established corrections, documented, tested against known cases (§17).
6. **Explanation rot.** Wrong advice is worse than no advice. Mitigation:
   observational-only, content-as-data, doc-linked, reviewed like docs.
7. **pandas/sklearn version chase.** Mitigation: narrow tested window, matrix
   CI, boundary wrapping.
8. **Hidden-state creep.** Convenience features that remember things
   invisibly. Mitigation: §12 — if it's remembered, it's a readable file.
9. **Solo-maintainer burnout.** Mitigation: verbs can be feature-frozen
   independently; extras isolate heavy deps; plugins move growth out of core;
   this document makes "no" cheap.
10. **LLM commoditization.** Agents can write one-off EDA code on demand.
    Michi's defensible ground is consistency, statistical correctness, and
    durable artifacts — an agent *should* call `michi bench --json` rather
    than reinvent honest CV. Mitigation: treat agents as a primary persona.

## 23. Version 2 roadmap (post-1.0; each item needs an ADR)

- `michi diff` — drift/shift report between two datasets or two runs.
- Optuna-backed HPO cells in `sweep` (extra).
- ONNX adapter (extra); Excel/SQL loaders if demand is proven.
- Pretrained/foundation-model adapters (e.g. TabPFN, hub models) behind an
  extra — downloads always explicit, cached locally, never required; the
  offline-first identity is non-negotiable.
- Recipe update flow: regenerate exported pipeline code when a recipe changes,
  with a reviewable diff.
- MLflow / W&B manifest exporters (interop, not competition).
- Time-series task support in `inspect`/`bench`.
- First-class agent affordances: machine-readable capability catalog,
  `--json` schema docs.

## 24. Version 3 horizon

- Text-task and tabular-DL benchmark packs — **as community plugins**, not core.
- Template/recipe registries maintained by the community.
- Optional, off-by-default local-LLM hook for conversational explanations —
  pluggable, never required, never a network call by default.
- The best v3 outcome is that core Michi is *boring*: frozen surfaces, plugin
  ecosystem carrying the growth, maintainer reviewing contracts instead of
  writing features.

**Never, at any version** (§7): AutoML, serving, monitoring, cloud, accounts,
telemetry, notebooks, DAG orchestration, a model registry, hidden sessions.

## 25. Long-term maintenance strategy

- **Growth goes to the edges.** After v1.0, new capability lands as plugins
  and template content, not core code. Core's job is stability.
- **Any verb can be feature-frozen independently** — the toolbox shape means
  maintenance can shrink to match maintainer capacity without breaking users.
- **Bus factor:** this file + ADRs + module READMEs are written so a new
  maintainer can take over from documents, not oral history.
- **Dependency hygiene:** core dep list reviewed each minor release; anything
  unused is removed; extras never migrate into core without an ADR.
- **Saying no is the primary maintenance tool.** §7 and §21 exist so refusals
  are citations, not arguments.
