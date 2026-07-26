# ADR-0001: Michi is a toolbox, not a workflow

**Status:** accepted (2026-07-26)

## Context

ML developer tools that own the user's workflow — required project
structures, wrapped training loops, session state, orchestrated stages —
consistently plateau: experienced users refuse the lock-in, and the wrapper
chases its underlying libraries' APIs forever. Tools that bolt onto existing
work with zero commitment (one-command profilers, linters, formatters) get
adopted. Michi's goal is daily use by practitioners on *their existing
projects*.

## Decision

Michi is a set of independent CLI verbs over durable artifacts (profile,
recipe, run manifest, report). Every verb:

- works standalone on artifacts the user already has (their CSV, their
  `model.pkl`, their repo), with no init step and no required layout;
- reads/writes files — there is no session state, no daemon, no database;
- is reachable non-interactively with flags and `--json`; interactive flows
  are authoring sugar that always emit an equivalent command and artifact.

Modules communicate **through artifacts**, never through a shared runtime
context object. The user's code is the pipeline; michi never wraps training
loops, never orchestrates stages, and never requires `import michi` in user
code.

## Consequences

- Adoption is per-verb and zero-commitment; deleting michi leaves the user's
  project fully working, plus useful files.
- Reproducibility is structural (any run re-executes from its artifacts).
- Some conveniences are deliberately forgone: no cross-command in-memory
  caching, no "current project" magic beyond the explicit `michi.toml`
  defaults file (PLAN.md §12).
- Features that require owning the workflow (AutoML, serving, monitoring,
  DAGs) are permanently out of scope (PLAN.md §7).
