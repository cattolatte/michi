# ADR-0003 — A guided sequence that is still a toolbox

- **Status:** Accepted
- **Date:** 2026-07-27
- **Supersedes:** nothing. ADR-0001 stands.

## Context

[ADR-0001](0001-toolbox-not-workflow.md) rules out workflow ownership: michi is
a set of independent verbs over artifacts the user already has, not a pipeline
that runs a project from end to end.

That decision has always had an unhappy side effect. A newcomer facing nine
verbs has no idea which one to type first, and "they are all independent" is a
true answer to a question nobody asked. The verbs *do* have a natural order —
not because michi has an opinion about the user's project, but because a recipe
must exist before it can be applied, and runs must exist before they can be
reported.

The obvious fix is a guided mode, and the obvious objection is that a guided
mode is a workflow. Both are half right. What makes a workflow tool a workflow
tool is not that it runs several things in sequence; it is that it decides
*what should happen* and reduces the user to approving it. A sequence that asks
at every stage, and never suggests an answer, automates the typing and nothing
else. That is the same line michi already draws inside `clean`, which is an
interactive wizard that asks a series of questions, writes a recipe, and prints
the command that would have produced it without the prompts.

## Decision

Add two console commands.

`path` prints the stages, the command that covers each, and a mark showing
which the current context could run right now. It executes nothing.

`walk` visits the stages one at a time. At each one it shows what the stage is,
what command covers it, and offers **run / skip / stop**.

A guided sequence is permitted only under all five of these constraints:

1. **It asks; it never suggests.** No stage may present a default that encodes
   a modelling judgement, rank the options, or mark one "recommended".
   Questions collect facts michi cannot know — which column is the target —
   never opinions michi should not hold.
2. **Every stage is optional, and the order is mechanical.** Any stage may be
   skipped and the walk left at any point. The order exists because of data
   dependencies, not importance.
3. **You can enter anywhere.** `walk bench` starts at the comparison stage.
   There is no prerequisite chain to satisfy first.
4. **Nothing runs that the user could not have typed.** Each stage prints the
   one-shot command, dispatches through the same Typer application as the
   shell, and records it in history — so a walk exports to a script exactly
   like a hand-typed session. This is the existing flag-parity rule; a walk is
   not allowed to be the only way to reach anything.
5. **A walk leaves no residue.** No resume file, no "stage 3 of 8" state, no
   hidden record of what was "completed". A walk that ended halfway is
   indistinguishable afterwards from someone who ran those commands by hand,
   because that is exactly what it was.

## Consequences

**What this buys.** The newcomer problem gets an answer that is not a tutorial
in the docs. The user who only wants one verb is unaffected: `path` shows the
stage they care about and they run it.

**What it costs.** A fifth surface to keep in flag parity. Every new verb now
needs a decision about whether it is a stage, and a stage entry if so.

**What would violate this ADR.** A `walk --auto` that runs every stage without
asking. A default that pre-selects "run" on stages michi thinks matter. Any
prompt phrased as "recommended:". Persisting walk progress to disk. Any of
these would need a new ADR, and would be a change to what michi is rather than
an addition to what it does.

**The test that enforces it.** Constraint 4 is checked mechanically: a test
asserts that every stage's command is a real verb of the CLI application, so a
stage can never be a console-only capability.
