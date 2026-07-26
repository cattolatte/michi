# Changelog

All notable changes to michi are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
