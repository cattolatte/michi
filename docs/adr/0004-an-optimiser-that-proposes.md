# ADR-0004 — An optimiser that proposes

- **Status:** Accepted
- **Date:** 2026-07-27
- **Relates to:** ADR-0001 (toolbox, not workflow), ADR-0003 (a guided sequence)

## Context

`michi tune` searches hyperparameters by random sampling, successive halving,
or exhaustive grid. All three draw from a space the user can print with
`--list-space` and replace with their own YAML. None of them has an opinion:
random sampling treats every configuration as equally worth trying, and grid
search tries all of them.

A model-based optimiser — Bayesian optimisation, TPE, whatever the library
calls it — does something different in kind. It fits a model of the objective
as it goes and *proposes* the next configuration from its own beliefs about
where the good ones are. It is, quite literally, a thing that forms opinions
and acts on them, inside a tool whose first principle is that it does not.

That deserved an ADR rather than a commit, because the surface argument
against it is strong and the surface argument for it is weak.

The weak argument for: "it finds better hyperparameters faster." True, and
insufficient — AutoML also finds better everything faster, and michi does not
do AutoML.

The argument that actually decides it is narrower. Ask what the optimiser has
an opinion *about*. It does not choose the model. It does not choose the
space. It does not choose the metric, decide when the result is good enough,
or say which configuration to ship. It chooses **the order in which
candidates from the user's own space are evaluated** — and that is the same
category of decision as which rows land in which fold, or how many bootstrap
resamples to draw. It is a mechanic. michi has always made mechanical choices
and has always been explicit that it does: "defaults exist for mechanics —
folds, seeds — never for judgement" (PLAN §3).

The line michi draws is between *how the work is done* and *what the work
means*. A sampler is the first. Nothing about a smarter sampler moves it into
the second.

There is, however, a real hazard, and it is not the one the surface argument
worries about.

## The hazard

A model-based optimiser is *better at overfitting the inner folds*. That is
not a side effect; it is the mechanism working as designed. Random search
wastes most of its budget on configurations that are nowhere near the
optimum. TPE spends its budget near whatever looked good on the inner split —
including where that split happened to be lucky.

So the gap between the search's own best score and the honest held-out score
grows with the sophistication of the optimiser. A tool that reports the inner
score gets *more* wrong the better its search gets, which is the worst
possible direction for an error to move.

michi already reports both numbers side by side, precisely so this is visible.
Bayesian search makes that existing decision load-bearing rather than merely
correct.

## Decision

Add `--strategy bayes` to `michi tune`, behind an optional extra, under five
constraints.

1. **The space stays printable and stays the user's.** `--list-space` works
   unchanged; `--space my.yaml` replaces it unchanged. The optimiser samples
   from that space and may not widen it, add a parameter, or continue outside
   its bounds.

2. **Nested scoring is not optional and not configurable.** The reported score
   comes from outer folds the search never touched. There is no flag to
   report the inner score as the result, because that flag would be a footgun
   whose only use is producing a number that flatters.

3. **The optimism gap is always shown.** Because it grows with the optimiser's
   strength, `tune` prints the inner best beside the held-out score for every
   strategy, and the doc states plainly that a *larger* gap under `bayes` is
   the optimiser working, not failing.

4. **Reproducible under a seed.** Same seed, same space, same data, same
   result. A search that cannot be repeated cannot be reviewed, and every
   other number michi produces is reproducible.

5. **Optional, and degrading to a named alternative.** The dependency is an
   extra. Without it, michi says which extra to install and which strategies
   work now — it does not silently fall back to random search, because a user
   who asked for `bayes` and got `random` would compare two runs that were
   never the same experiment.

## Consequences

**What this buys.** A budget of thirty evaluations goes much further on a
space of any size, which is the common case for a gradient-boosting model with
six interacting parameters. For a hackathon or a competition — where compute
is the binding constraint and the space is large — this is the difference
between tuning and not bothering.

**What it costs.** A fourth strategy to document and keep honest, and an
optional dependency. More importantly, it raises the stakes on the nested
scoring michi already does: with a weaker search the inner/outer gap was a
teaching point, and with this one it is a guardrail.

**What would violate this ADR.** An optimiser that proposes a model rather
than a configuration. A search that expands the user's space "because the
optimum is at the boundary". Reporting the inner score as the result under any
flag. Silent fallback to a different strategy. Any of these turns a sampler
into something that decides, and would need a new ADR — and a different
answer.
