# ADR-0002: Freeze the public surface at 1.0

**Status:** accepted (2026-07-27)

## Context

michi reached v0.8 with four artifact types, nine verbs, and two extension
points. All of them have now been exercised by real use across eight
milestones: the profile schema has survived every finding detector added since
v0.1; the run manifest has carried evaluations, benchmark cells, and sweep
cells without a shape change since v0.2; the recipe has been written,
hand-edited, applied, and compiled to code since v0.4.

Users are beginning to build on these: manifests get committed to
repositories, recipes get reviewed in pull requests, and plugin authors need
to know what they can rely on. Continuing to reserve the right to change any
of it makes michi unusable for exactly the audience it was built for.

Freezing too early carries the opposite risk — carrying a mistake forever —
which is why this happens at 1.0 and not at 0.2.

## Decision

From 1.0, the following are public API under semantic versioning. A breaking
change to any of them requires a major version.

**Artifact schemas** — profile (1.0), run manifest (1.0), recipe (1.0). Every
artifact carries its `schema_version`; michi reads any artifact whose major
version it knows. New fields may be added (they are additive and optional);
existing fields may not change meaning, type, or disappear.

**The CLI surface** — the names of verbs, their flags, and the meaning of
their exit codes. Output *formatting* is not frozen: tables may be laid out
differently, and colours may change.

**The plugin contract** — the `michi.models` and `michi.adapters` entry-point
groups, the `ModelEntry` fields michi reads, and the adapter protocol
(`handles`, `load`).

**The Python API** exposed in each package's `__all__`. Everything else is
private, whatever its visibility.

Not frozen: terminal layout, HTML and report styling, explanation prose,
finding thresholds (these are named and documented, but tuning them is a
patch), and the model catalogue's contents beyond the guarantee that a
documented name keeps its meaning.

## Consequences

- Reading a michi artifact written today will keep working. A regression test
  reads committed 1.0 fixtures of all three artifact types on every CI run,
  so this is enforced rather than promised.
- Adding a field to an artifact is a minor release; removing or repurposing
  one is a major release, which in practice means it will not happen.
- Mistakes now cost. The thresholds, metric names, and step vocabulary chosen
  in 0.1–0.8 are what michi lives with.
- The freeze is what makes the toolbox claim real: a verb you adopt today
  behaves the same next year, which is the only basis on which someone puts
  michi in a pipeline.
