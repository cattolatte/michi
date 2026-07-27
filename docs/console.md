# The console

Run `michi` with no arguments.

```
  ███╗   ███╗██╗ ██████╗██╗  ██╗██╗
  ████╗ ████║██║██╔════╝██║  ██║██║
  ██╔████╔██║██║██║     ███████║██║      道
  ██║╚██╔╝██║██║██║     ██╔══██║██║
  ██║ ╚═╝ ██║██║╚██████╗██║  ██║██║
  ╚═╝     ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝

        =[ michi v1.0.2  ·  a local-first ML workbench                     ]
+ -- --=[ 9 verbs  ·  14 models  ·  7 recipe ops  ·  26 explanations       ]
+ -- --=[ 19 console commands  ·  no account, no telemetry, no network call]

  心得  michi lists the options; you choose.
  path walks the stages · help lists commands · use <file> loads data

  tip: `bench` always includes a dummy baseline, so "is this any good?"
       has an answer instead of a vibe.

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

**道 the path** — `path`, `walk [stage]`

**Context** — `use <path>`, `set <key> <value>`, `unset <key>`,
`show [context|columns|models|runs]`

**Verbs** — `inspect`, `eval`, `bench`, `clean`, `apply`, `export`, `sweep`,
`report`. Every flag of the shell command works unchanged.

**Session** — `help [command]`, `history [--export <file>]`, `save [path]`,
`clear`, `exit` (or `quit`, or Ctrl-D)

## The path

Nine verbs with no stated order is a fair description of a toolbox and a poor
answer to "where do I start?". `path` prints the stages of a tabular project,
the command that covers each, and a ✓ on the ones your current context could
run right now.

```
michi (customers.csv → purchased) › path

  ✓  見  miru — see what you have             inspect   inspect --explain
  ✓  整  totonoeru — decide what to fix       clean     clean --target <col>
  ·  直  naosu — produce the fixed data       apply     apply -o clean.parquet
  ✓  比  kuraberu — compare models honestly   bench     bench --models …
  ✓  確  tashikameru — verify one model       eval      eval <model.pkl>
  ✓  探  sagasu — search a grid               sweep     sweep sweep.yaml
  ·  記  shirusu — write up what happened     report    report runs/
  ·  出  dasu — take the code and go          export    export -o pipeline.py

  ✓ marks a stage this context can run right now. Every stage stands alone —
  run one and stop, or none of them. michi walks none of it for you.
```

`path` executes nothing. It is a map.

## Walking it

`walk` visits the stages one at a time and **asks before each one**:

```
michi › walk

 道   michi walk  ·  8 stages

  Every stage asks before it runs, and every stage is optional.
  michi never picks for you — it only does what you say, and prints
  the command it ran so you can run it yourself next time.

  [1/8]  見 miru — see what you have
      covered by michi inspect --explain
  ? What would you like to do?  (Use arrow keys)
    run inspect
  » skip this stage
    stop walking
```

Every stage offers **run / skip / stop**. The prompts collect facts michi
cannot know — which column is the target — and never opinions michi should not
hold. Nothing is marked "recommended", and the default is *skip*.

`walk bench` starts at the comparison stage; there is no prerequisite chain to
satisfy first. Each stage prints the one-shot command before running it and
records it in history, so a walk exports to a script exactly like a hand-typed
session.

There is no resume file and no "stage 3 of 8" state on disk. A walk you left
halfway is afterwards indistinguishable from someone who ran those commands by
hand — because that is exactly what it was.

The five constraints that keep a guided sequence from becoming a workflow tool
are recorded in
[ADR-0003](adr/0003-a-guided-sequence-that-is-still-a-toolbox.md).

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
