# ADR-0005 — Freeze the expanded surface

- **Status:** Accepted
- **Date:** 2026-07-27
- **Extends:** [ADR-0002](0002-freeze-the-public-surface.md)

## Context

ADR-0002 froze michi's public surface at 1.0: the artifact schemas, the CLI
verbs and their flags, the plugin contract, and each package's `__all__`. It
was written when there were nine verbs.

There are now seventeen. `diff`, `ensemble`, `errors`, `fit`, `predict`,
`split`, `threshold`, and `tune` all arrived after the freeze, along with ten
new recipe operations, two neural-network models, and new keys in the run
manifest.

**None of them are covered.** ADR-0002 promises stability for a list that no
longer describes the tool. Someone building a pipeline on `michi predict` today
has no stated guarantee, while someone building on `michi eval` does — and
nothing in the documentation explains why one is safer than the other. That is
a worse position than either freezing everything or freezing nothing, because
it is invisible.

Meanwhile the Python API has been frozen since 1.0 and never documented as
something to use. A guarantee nobody knows about buys nothing.

## Decision

Extend the ADR-0002 freeze to everything michi now exposes. From 2.0:

**All seventeen CLI verbs** are public API — their names, their flags, and the
meaning of their exit codes. Output *formatting* remains unfrozen: tables may
be laid out differently and colours may change, because a renderer that can
never change is a renderer that can never improve.

**All seventeen recipe operations** and their parameter names. The recipe
schema version stays at 1.0 — the shape never changed, only the vocabulary
grew — and a recipe written today will load in every 2.x michi.

**The extended run manifest**, including `details.importance`, `oof`, and the
per-slice and calibration keys added since 1.0. New keys may still be added;
existing ones may not change meaning or disappear.

**Two new plugin groups**: `michi.metrics` alongside the existing
`michi.models` and `michi.adapters`. A metric plugin supplies a callable and
an optional `greater_is_better` attribute.

**The Python API**, now documented in [api.md](../api.md) rather than merely
promised. Each package's `__all__` is public; everything else is private
whatever its visibility.

## Why this is a major version

Nothing breaks. No verb is renamed, no flag removed, no schema reshaped —
every 1.x command runs unchanged on 2.0.

Semantic versioning is about the *contract*, and this changes it: michi is
committing to keep stable a surface roughly twice the size of the one it
committed to at 1.0. A user who reads "2.0" should understand that the whole
toolbox now carries the guarantee, not the third of it that happened to exist
in July. Announcing that as a patch would be underselling a real promise;
announcing it as 1.12 would leave the promise looking like a footnote.

The alternative — leaving eight verbs permanently outside the freeze — was
rejected because it makes the guarantee unusable. A contract you have to check
per-verb is not a contract.

## Consequences

**Mistakes now cost more.** `--group`, `--metric`, `--oof`, `--calibrate`, and
the text and time operations were all designed in the weeks before this ADR.
They are what michi lives with. That is why group-aware cross-validation was
fixed *before* this freeze rather than after: `bench` ignoring groups was a
defect, and freezing a defect makes it a feature.

**The plugin surface grows.** `michi.metrics` is now a contract, which means a
metric plugin written today keeps working — and that michi cannot change what
it passes a scorer without a major version.

**Growth continues at the edges.** The core is now large enough. New metrics,
new models, and new loaders arrive as plugins; new verbs need a reason strong
enough to justify a permanent commitment.

**What would force 3.0.** Reshaping an artifact schema, removing or renaming a
verb, changing what a flag means, or dropping a Python version. In practice
that means it will not happen soon, which is the point.
