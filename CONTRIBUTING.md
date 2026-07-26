# Contributing to michi

Thank you for looking. This file exists so that "no" is a citation rather than
an argument, and so that a yes is easy.

## Before you start

Read the [philosophy](docs/philosophy.md). michi is a **toolbox, not a
workflow**, and it **automates implementation, never judgement**. A pull
request that conflicts with either is declined regardless of how good the code
is — that is not a comment on the work, only on the fit.

In particular, michi will never gain: AutoML or automatic model selection, a
required project structure, a training-loop wrapper, cloud features, accounts,
telemetry, serving, monitoring, DAG orchestration, or hidden session state.

## What is most welcome

- **Bug reports with a dataset shape that reproduces them.** Tiny CSVs are
  perfect.
- **Findings and checks michi should raise but does not** — with a sentence on
  what the finding means and what options a practitioner has.
- **Corrections to explanation content** in `src/michi/explain/content/`. This
  is prose, reviewed like documentation, and accuracy matters more than
  anything else in the repository.
- **Cross-platform fixes.** Windows is first-class; if something is awkward
  there, that is a bug.
- **Plugins.** See [the plugin guide](docs/plugins.md) — new model families and
  loaders belong in your package, not michi's.

## Working on it

```bash
uv sync --extra dev --extra ui
uv run ruff check . && uv run ruff format --check .
uv run mypy src/michi
uv run pytest
```

All four must pass. CI runs them on Linux, macOS, and Windows against Python
3.11 and 3.13.

## House rules

- **One slice per pull request**: a verb, a finding, a fix. Tests and
  documentation land with the change, not after it.
- **Concrete before abstract.** An interface needs two real implementations
  before it exists. Cite them in the description.
- **New dependencies need a reason** in the pull request; new *core*
  dependencies need an ADR in `docs/adr/`.
- **Tests assert behaviour, never exact floats**: shapes, invariants, "beats
  the dummy baseline", "is not called significant". A test that pins a number
  breaks on the next sklearn release and teaches nobody anything.
- **Every module docstring states what the module deliberately does not do.**
  That sentence is often the most useful one in the file.
- **Error messages are actionable**: name the exact command to run next.

## Style

Python 3.11+, `from __future__ import annotations`, full type hints, mypy
strict, ruff for lint and formatting, line length 88. Frozen dataclasses for
value objects. Errors raised from the `MichiError` hierarchy at public
boundaries, with third-party failures wrapped and chained.

Comments explain *why*, not *what*. If a threshold is arbitrary, say so and
say what it trades off.

## Reviews

Reviews are about the code and the fit, never the person. A "no" on scope is
not a judgement of quality — michi says no to most things on purpose, and that
is what keeps it maintainable by a small number of people for a long time.
