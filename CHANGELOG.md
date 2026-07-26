# Changelog

All notable changes to michi are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
