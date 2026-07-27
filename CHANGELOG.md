# Changelog

All notable changes to michi are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`michi.toml` now reaches every command**, not only the console. The
  documented precedence — flags > `michi.toml` > built-in — was previously
  true only inside the console, which made the documentation wrong for every
  shell user. `michi inspect`, `bench`, `clean`, and `report` now run with no
  arguments in a configured project.
- **`--recipe` on `eval`**, which the documentation had promised. Only the
  recipe's deterministic steps run: imputers in a recipe were fitted for
  training, and re-fitting them on the evaluation set would quietly change
  what is being measured.
- A PyPI publishing workflow using trusted publishing, which re-runs every
  gate against the tagged commit and refuses to publish if the tag and the
  package version disagree.

### Changed
- The distribution is **`komichi`** on PyPI, because `michi` was already
  taken. The command and the import are both still `michi`.
- A *configured* target is a hint, not a command: running against data that
  lacks it prints a note and continues, rather than failing a run the user did
  not misconfigure. A typed `--target` that is missing still fails loudly.

### Fixed
- Error messages naming an extra were rendered as terminal markup, so
  `pip install 'komichi[bench]'` reached users as `pip install 'komichi'` —
  an install command that silently does the wrong thing. Messages are now
  rendered verbatim, and a test asserts the brackets survive.
- The install command is built from one constant, and a test fails if any
  message hardcodes the package name — which immediately caught a stale hint
  in the Excel loader.

## [1.0.0] — 2026-07-27

The public surface is frozen.

### Added
- **A quickstart** taking one messy CSV to a compared, reported model in
  fifteen minutes — the page that decides whether michi is worth adopting.
- **[ADR-0002](docs/adr/0002-freeze-the-public-surface.md)** records exactly
  what is now public API under semantic versioning: the profile, run manifest,
  and recipe schemas (all 1.0); the verb names, flags, and exit codes; the
  plugin entry-point groups and contract; and each package's `__all__`.
  Terminal layout, report styling, explanation prose, and finding thresholds
  are deliberately *not* frozen.
- **Schema-stability tests** that read committed 1.0 fixtures of all three
  artifact types on every CI run, execute every frozen recipe operation, and
  compile a 1.0 recipe to code. A promise nobody tests is a hope.

### Fixed
- A recipe step's `why` — the reason a column was dropped — was written as a
  YAML comment and silently lost on reload. It is now a field and round-trips.
  That reason is exactly the part nobody can reconstruct six months later, and
  the freeze tests are what caught it.

### Notes
- The freeze is what makes the toolbox claim real: a verb adopted today
  behaves the same next year, which is the only basis on which anyone puts
  michi in a pipeline.
- Everything on the roadmap has now shipped: `inspect`, `eval`, `bench`,
  `report`, `clean`/`apply`/`export`, the console, `sweep`, `ui`, and plugins.

## [0.8.0] — 2026-07-27

The plugin surface.

### Added
- **Two extension points**, discovered via entry points:
  `michi.models` adds algorithms to the `bench` catalogue, and
  `michi.adapters` adds ways to load a model for `eval` — which is where an
  ONNX or framework loader belongs, since michi deliberately ships none.
- **`michi plugins`** lists what is installed, working or not.
- **A published compatibility suite** (`michi.plugins.check_model_entry`,
  `check_adapter`) that plugin authors run in their own CI. michi's maintainer
  is not the integration point for every plugin — that is the only way an
  ecosystem stays affordable for one person to support.
- `CONTRIBUTING.md` and a plugin guide.

### Notes
- These points were opened at v0.8, not before, because the interfaces behind
  them had by then survived six milestones and a dozen concrete
  implementations without changing shape. An interface with one implementation
  is a guess, and a published interface is a promise.
- **A broken plugin is skipped, never fatal**: an entry point that fails to
  import is reported by `michi plugins` and ignored, and benchmarks keep
  running.
- **Built-ins win ties.** A plugin cannot shadow `rf` or any documented name,
  so `--list-models` always means what the documentation says.
- Recipe operations, report sections, and metric definitions remain closed.
  They stay closed until something real needs them.

## [0.7.0] — 2026-07-27

A local viewer: `michi ui`.

### Added
- **`michi ui`** (needs `michi[ui]`) — a read-only web view over the runs
  directory: every run grouped by dataset and target, and a detail page per
  run with metrics beside their baselines, checks, slices, and full
  provenance.
- Serves on **localhost only**, with no flag to change that.

### Notes
- The viewer is **read-only by construction**: a test asserts the application
  exposes only `GET`. A UI that can act is a platform, and michi is not one.
- **No database, no build step, no network.** Every request reads the
  directory, so a run appears on refresh; pages are server-rendered HTML with
  inline CSS, so the viewer works air-gapped and cannot rot when a frontend
  toolchain moves on.
- **It is deletable.** Removing it would remove convenience and not one
  capability — everything it shows is a file, and `michi report` renders the
  same artifacts. That is the bar it has to keep clearing.

## [0.6.0] — 2026-07-27

Experiment grids: `michi sweep`.

### Added
- **`michi sweep`** — executes a declarative grid of models × recipes × seeds
  from a YAML plan, recording one run manifest per cell. Progress is printed
  as it goes, and `--dry-run` lists the grid without running anything.
- **Content-hash caching and resumption.** Each cell's identity is a hash of
  the data, recipe, model, seed, and fold count that produce it, so an
  interrupted sweep resumes exactly where it stopped, and editing one recipe
  re-runs exactly the cells that use it. A long sweep *will* be interrupted;
  resumption is not a nicety.
- **Contained failure**: a cell that cannot train is recorded and the grid
  continues, because losing thirty completed cells to the thirty-first would
  be indefensible.
- **`--recipe` support in `bench`** — a recipe's deterministic steps run once
  up front; its fitted steps become a transformer fitted inside each fold, so
  comparing preprocessing strategies stays leak-free. A recipe the user wrote
  takes precedence over michi's default preparation.
- Sweep manifests feed straight into `michi report`.
- Documentation for `sweep`.

## [0.5.0] — 2026-07-27

The console: `michi` with no arguments.

### Added
- **An interactive console** with the context always visible in the prompt
  (`michi (train.csv → churned) ›`), `help`, `use`, `set`, `unset`, and
  `show [context|columns|models|runs]`.
- **Tab completion over your own data** — once a dataset is loaded, `set
  target <TAB>` completes that file's column names. This is the one thing a
  one-shot CLI can never do, and the reason the console exists.
- **`history --export`** writes the session as a replayable shell script of
  fully-expanded one-shot commands. Console-only commands never appear, so
  exploration stays reproducible.
- **`michi.toml`** — optional project defaults, found by searching upward from
  the working directory. Precedence is flags > `michi.toml` > built-in, and
  `michi info` prints which file was found and what it supplies. This is
  michi's answer to session state: if it is remembered, it is a file you can
  read, edit, diff, and commit.
- The console's context *is* the `michi.toml` model, so `save` persists it and
  a later session restores it.

### Fixed
- Console line splitting treated a backslash as an escape character, which
  silently mangled every Windows path a user typed. Backslashes are now
  literal; quoting still groups arguments containing spaces.

### Notes
- The console contains **zero logic**: every verb dispatches to the same Typer
  application the shell invokes, and a test asserts no console verb exists
  outside the CLI. Deleting the console would remove convenience, never
  capability.
- Bare `michi` prints help rather than opening a prompt when stdout is not a
  terminal, so scripts never hang.

## [0.4.0] — 2026-07-26

Cleaning decisions as a file you own: `michi clean`, `apply`, and `export`.

### Added
- **`michi clean`** — authors a **recipe**: an ordered list of declarative
  cleaning operations plus the schema they were written against. Your data is
  never touched. The interactive mode walks the *findings* `inspect` produced,
  grouped, so a two-hundred-column dataset yields a handful of questions
  rather than two hundred prompts — and "leave them as they are" is always
  present and always the default.
- **Full flag parity**: `--drop`, `--dedupe`, `--cast`, `--impute`, `--clip`,
  `--encode`, `--scale` express everything the wizard can, and every session
  prints the non-interactive command that reproduces it.
- **`michi apply`** — executes a recipe non-destructively to a new file. The
  recipe's schema snapshot acts as a data contract: applying it to data that
  lacks the named columns fails loudly, with `--no-strict` to degrade
  gracefully.
- **`michi export`** — compiles a recipe into a standalone Python module that
  imports pandas and scikit-learn, never michi. Deterministic steps become a
  `prepare()` function; fitted steps become an sklearn transformer to fit
  inside cross-validation. Generated code passes ruff check and format, and a
  test executes it and asserts it matches `michi apply`.
- **Recipe artifact** (schema 1.0) written as commented YAML through a
  template — `yaml.dump` cannot produce a comment, and the comments carry the
  reason each decision was made.
- Honest leakage reporting: steps that learn from data are marked as such,
  `apply` says when it fitted them on a whole file, and the caveat is written
  into the recipe itself.
- Documentation for `clean`, `apply`, and `export`.

## [0.3.0] — 2026-07-26

Honest model comparison: `michi bench` and `michi report`. This completes the
MVP — inspect, evaluate, compare, report.

### Added
- **`michi bench`** — cross-validates several models and reports which of them
  are actually distinguishable. A dummy baseline is always added; differences
  from the leader are tested with the **corrected resampled *t*-test**
  (Nadeau & Bengio, 2003) and Holm-adjusted across models, and the conclusion
  is stated in plain language ("tied with leader at this sample size").
- **A model catalogue** (`--list-models`) of 14 algorithms: `dummy`, `linear`,
  `ridge`, `lasso`, `tree`, `rf`, `extra-trees`, `hist-gbm`, `knn`, `svm`,
  `naive-bayes` in the base install, plus `xgb`, `lgbm`, `catboost` behind the
  `bench` extra. Everything trains locally; nothing is downloaded.
- **Stated column preparation**, printed every run and recorded in every
  manifest, and fitted *inside each fold* so a benchmark cannot leak.
  Overridable with `--impute`, `--encode`, `--no-scale`.
- **`michi report`** — renders recorded runs as a self-contained offline HTML
  page, Markdown for a pull request, or a paste-ready booktabs LaTeX table for
  a paper. Runs are grouped by dataset content hash and target, so
  incomparable numbers are never put in one table.
- Benchmark checks: `below-baseline`, `no-clear-winner`, `model-failed`.
- Documentation for `bench` and `report`.

### Changed
- Unknown or task-incompatible model names now fail immediately, before any
  training, rather than appearing as a failed row in the leaderboard.

### Fixed
- The corrected *t*-test returned "not significant" when two models differed
  by exactly the same amount on every fold. A zero-variance difference is the
  strongest possible evidence, not the weakest; the limit is now p = 0.

## [0.2.0] — 2026-07-26

Evaluate models michi never saw trained: `michi eval`.

### Added
- **`michi eval`** — evaluate an existing model against a dataset. Reports
  metrics with 95% bootstrap confidence intervals, always beside trivial
  baselines, plus a confusion matrix, per-slice scores, and a calibration
  curve.
- **Two ways to supply a model**: sklearn-compatible pickle/joblib files, or
  *any* object exposing `predict(X)` via a `mymodule:my_model` reference —
  which covers PyTorch, TensorFlow, ONNX, and bespoke models without michi
  shipping a loader per framework. Formats that cannot be loaded honestly are
  refused with a route forward rather than a confusing failure.
- **Eight evaluation checks**: `below-baseline`, `beats-baseline`,
  `suspiciously-perfect`, `single-class-predictions`, `wide-interval`,
  `miscalibrated`, `slice-gap`, and `small-evaluation-set` — each with an
  explanation and options, never a recommendation.
- **Run manifests** (schema 1.0) written to `runs/` by default: dataset and
  model hashes, seed, metrics, baselines, checks, and the environment,
  making any number traceable to the conditions that produced it.
- **CI gate**: `--fail-under metric=value`, respecting each metric's
  direction so `rmse=3.0` passes when RMSE is at most 3.0.
- Automatic task detection (classification or regression), overridable with
  `--task`.
- Documentation for `eval`, including the predict protocol.

### Changed
- sklearn's internal warnings are suppressed during evaluation; michi reports
  the same conditions itself, as checks, in language a user can act on.

## [0.1.0] — 2026-07-26

The first working verb: `michi inspect`.

### Added
- **`michi inspect`** — profile any CSV, TSV, or Parquet file (Excel via the
  `michi[excel]` extra). Reports column kinds, missing values, duplicates,
  cardinality, distribution shape, and outliers, with findings ranked by
  severity.
- **Fifteen finding detectors**, including empty and constant columns,
  identifier-like and high-cardinality columns, skew, outliers, duplicate and
  near-perfectly-correlated columns, numbers and dates stored as text, mixed
  types, and — when `--target` is given — class imbalance and target-leakage
  suspects.
- **Explanations** (`--explain`) attached to every finding: what it means and
  which options practitioners choose between, never a recommendation.
  Explanation text lives in reviewable content files.
- **Three output surfaces**: a rich terminal report, a self-contained offline
  HTML report (`--html`, no CDN or JavaScript), and the versioned profile
  artifact as JSON (`--json`).
- **Honest sampling** for large files: seeded random sampling above 256 MB,
  always recorded in the artifact and the report, with `--full` to override.
- **CI gate mode**: `--fail-on high|warn|info` with meaningful exit codes.
- Artifact foundations: immutable, versioned value objects (profile schema
  1.0), SHA-256 content hashing, and the tabular io layer.
- Documentation for `inspect`, plus committed example artifacts in
  `examples/`.

### Notes
- Profile schema is versioned but not yet frozen; it comes under semantic
  versioning at 1.0.

## [0.0.1] — 2026-07-26

### Added
- Project skeleton: packaging, CI, module layout, error hierarchy, CLI entry
  point (`michi --version`, `michi info`), philosophy and roadmap
  documentation, ADR-0001.
