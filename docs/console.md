# The console

Run `michi` with no arguments.

```
        道  michi v0.5.0

        a local-first ML workbench — automate implementation,
        never judgement

        `help` lists commands · `use <file>` loads data ·
        Tab completes · Ctrl-D exits

michi › use data/customers.csv
loaded customers.csv — 13 columns
michi (customers.csv) › set target purchased
target = purchased
michi (customers.csv → purchased) › inspect
…
michi (customers.csv → purchased) › bench --models linear,rf
…
michi (customers.csv → purchased) › history --export session.sh
  wrote session.sh — a replayable script of one-shot michi commands
```

## What it is, and what it deliberately is not

The console is a **skin over the one-shot CLI**. Every verb builds an argument
list and hands it to the same application `michi bench` invokes. There is no
second code path, so the console can never behave differently from the command
line — and no capability can exist only here.

Deleting the console from michi would remove convenience. It would not remove
a single thing you can do.

## Why enter it at all

**Tab completion of *your* columns.** Once a dataset is loaded, `set target
<TAB>` completes the column names in that file. A one-shot CLI can never do
this, and it is the reason the console exists.

Completion covers commands, settings, `show` targets, model names, file paths,
and common flags.

## Context

The prompt shows your context at every keystroke: `michi (customers.csv →
purchased) ›`. Verbs inherit it, so `bench` expands to the full command.

```
michi › show context

  setting    value
  ─────────────────────────────
  data       customers.csv
  target     purchased
  recipe     —
  runs_dir   runs
  models     linear,rf,hist-gbm
  seed       0
  cv         5

  unsaved — `save` writes these to michi.toml
```

**Context is the `michi.toml` model, held in memory.** `set` edits it, `save`
writes it to disk, and a later session restores it. Nothing michi remembers is
invisible — that is the whole reason there is no hidden session state.

An explicit flag always beats the context:

```
michi (customers.csv → purchased) › bench --target something_else
```

## Commands

**Context** — `use <path>`, `set <key> <value>`, `unset <key>`,
`show [context|columns|models|runs]`

**Verbs** — `inspect`, `eval`, `bench`, `clean`, `apply`, `export`, `report`.
Every flag of the shell command works unchanged.

**Session** — `help [command]`, `history [--export <file>]`, `save [path]`,
`clear`, `exit` (or `quit`, or Ctrl-D)

## Exporting a session

```
michi (customers.csv → purchased) › history --export session.sh
```

```bash
#!/usr/bin/env bash
# Recorded from a michi console session.
# Every line is a plain one-shot command — the console adds nothing.
set -euo pipefail

michi inspect customers.csv --target purchased --seed 0
michi bench customers.csv --target purchased --runs-dir runs --seed 0 --cv 5 --models linear,rf
```

Console-only commands (`set`, `show`) never appear — only the verbs, fully
expanded, exactly as you would type them in a shell. Exploration stays
reproducible.

## `michi.toml`

`save` writes the context as a project defaults file:

```toml
# michi project defaults.
#
# These are default values for command-line flags, nothing more.
# An explicit flag always wins, and `michi info` shows where each
# value came from. Delete this file and michi behaves normally.

[defaults]
data = "data/customers.csv"
target = "purchased"
runs_dir = "runs"
seed = 0
cv = 5
```

It is found by searching upward from the working directory, so it works from a
subdirectory. Check it into git: it is how a team shares "the target column is
`purchased`" without anyone retyping it.

**Every command honours it**, not only the console:

```bash
michi inspect          # uses data + target from michi.toml
michi bench --no-save  # also uses models, cv, seed
michi report           # uses runs_dir
```

An explicit flag always wins. A configured *target* is treated as a hint
rather than a command: if you run against a different dataset that lacks it,
michi says so and continues without one instead of failing a run you did not
misconfigure.

`michi info` prints which file was found and what it supplies.

## Non-interactive contexts

Piped or redirected, bare `michi` prints help instead of opening a prompt, so
scripts never hang waiting for input.
