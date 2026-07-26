# Philosophy

Michi has one rule from which every other decision follows:

> **Automate implementation. Never automate judgement.**

Tools observe, report, and offer options. You decide. Michi then executes what
you chose and records it in a file you own.

## Principles

**Toolbox, not workflow.** Every verb is independently useful. There is no
project template to adopt, no required directory layout, no init step, and no
order you must follow. Use `michi inspect` and nothing else, forever, and the
tool has done its job.

**Menus, not recommendations.** Where a judgement call exists, michi lists the
options with factual one-line descriptions and you pick. Menus are contextual,
never encyclopedic: imputation options appear only for columns that actually
have missing values. Defaults exist for mechanics — cross-validation folds,
random seeds — never for judgement.

**Artifacts over sessions.** Every interactive decision compiles to a durable,
versionable file: a profile, a recipe, a run manifest, a report. The session is
the authoring interface; the artifact is the product. Nothing important lives
only in michi's memory.

**Rigor by default.** Dummy baselines, confidence intervals, significance
tests, and leakage checks are opt-out, not opt-in. A comparison that cannot
distinguish two models should say so in plain language.

**Transparency over convenience.** When the two conflict, transparency wins.
Michi shows its work, names its thresholds, and generates readable code rather
than hiding logic behind an abstraction.

**Local-first.** No server, no accounts, no telemetry, no network calls. Your
data never leaves your machine. Anything michi remembers is a plain file you
can read, edit, diff, and delete.

**Explanation through observation.** Findings state what is true about your
data — "this column is 77% missing" — and the explanation layer describes what
that means and which options exist. Michi never says "we recommend X", because
the recommendation is yours to make.

## Non-goals

These are permanent. Each would turn michi into a product that already exists,
or one that structurally fails its own philosophy.

| Not this | Why |
|---|---|
| **AutoML** | Michi never searches model space uninvited or declares a winner. It runs the comparisons you specify and reports honestly. AutoGluon and FLAML do the other thing well. |
| **Workflow ownership** | No required project structure, no DAG engine, no orchestrator. Your code is the pipeline. |
| **A training-loop wrapper** | You never write `import michi` inside your training code. Wrappers are leaky abstractions chained to their underlying library's release cycle. |
| **Cloud, accounts, telemetry** | Trust is a feature. Local-first is the identity. |
| **Serving and monitoring** | Evaluating a model on a dataset is in scope; watching live traffic is a different product. |
| **A notebook or IDE** | Michi complements notebooks; it does not replace them. |
| **Hidden session state** | Convenience that michi remembers invisibly breaks reproducibility, scripting, and CI. Defaults live in a readable `michi.toml`. |

If a proposed feature's pitch begins "michi manages X for you" — where X is
your code, your data decisions, or your modeling judgement — the answer is no.
