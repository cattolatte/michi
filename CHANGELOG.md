# Changelog

All notable changes to michi are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.10.0] — 2026-07-27

Three questions michi could not answer, all of which someone was answering by
hand.

### Added
- **`michi diff`** — has this data changed? Compares two datasets, or a
  dataset against a committed `profile.json`, reporting removed and added
  columns, type changes, missingness, distribution shift, new categories, and
  row-count collapse.

  The profile artifact was always half of this: `inspect` has written schemas
  and distributions since v0.1, so a drift check needs no new measurement —
  which is why it belongs in michi rather than in a monitoring service michi
  would have to run. Severity follows *breakage*, not effect size: a removed
  column is high whatever the distributions did; a new column is information,
  because a fitted model never asked for it. Shift is measured in baseline
  standard deviations so the threshold means the same thing for dollars and
  millimetres. `--fail-on high` makes it a nightly CI gate.

- **`michi threshold`** — the decision cutoff nobody chose. A classifier emits
  a probability and almost every tool turns it into a label at 0.5 silently.
  On imbalanced data, or when a miss costs more than a false alarm, that is
  simply the wrong number.

  michi prints precision, recall, F1, the errors, and the cost at every
  cutoff, and marks the best one *for an objective the user named*. With
  `--cost fn=10,fp=1` on the example dataset the best cutoff is **0.16, not
  0.50** — and 0.50 costs 195 against 79. Which trade to take depends on what
  a miss costs, which michi has no way to know, so it has no default beyond
  weighing the two errors equally.

- **`michi errors`** — the rows behind the score. Lists the mistakes ordered
  by how confident the model was, because a confident error is a mislabelled
  row, a leak, or a region the features do not describe, while an unsure one
  is the model behaving correctly at the edge. Reports the subgroups where
  errors concentrate, and writes every mistake to a file with `--out`.
  Regression targets are ranked by residual instead.

## [1.9.0] — 2026-07-27

### Added
- **`michi split`** — hold data out the way the data requires. A classification
  target is stratified without being asked; `--group` keeps rows sharing an
  entity on one side; `--time` holds out the future.

  The last two exist because a random split *lies*. Two rows from the same
  customer put four in training and one in test, and the score that comes back
  is memory rather than generalisation. A test set drawn from before the
  training rows asks the model to predict the past. Both need a column michi
  cannot guess, so both are flags — and a random split now says plainly what
  it cannot promise.

  The summary states the property the split was chosen to provide and verifies
  it: a grouped split confirms no group spans both sides, and says so loudly
  if one does rather than promising silently.
- **`eval --importance`** — rank columns by what the model loses when each is
  shuffled. Measured through `predict` alone, so a PyTorch network and a
  random forest are measured identically and michi needs no per-model
  introspection to keep working.

  Each column is shuffled several times and the spread reported, because an
  importance smaller than its own noise is not a finding — those rows are
  marked "within noise" and drawn faintly in the viewer. The output states
  what it is: *what this model uses*, not what matters. A column the model
  ignores may still drive the outcome, and two correlated columns split the
  credit between them, which has misled people into deleting a feature that
  mattered.
- A column-importance chart in `michi ui`, and `分 split` in the CLI stage map.

### Fixed
- `eval --importance` was declared and never passed through to the evaluator,
  so the flag silently did nothing. Caught by running it.

## [1.8.0] — 2026-07-27

### Added
- **`tune --strategy bayes`** (`pip install 'komichi[bayes]'`) — model-based
  hyperparameter search, proposing each configuration from what it has learned
  rather than drawing blind.

  [ADR-0004](docs/adr/0004-an-optimiser-that-proposes.md) records why an
  optimiser that forms opinions belongs in a tool that does not: it has an
  opinion about *the order candidates from the user's own space are tried in*,
  which is the same category of decision as fold assignment — a mechanic, not
  a judgement. It never chooses the model, the space, the metric, or what to
  ship.

  The real hazard is the opposite of the obvious one. A stronger optimiser is
  *better at overfitting the inner folds*, so the gap between its own best
  score and the honest held-out one grows with its sophistication. Measured on
  the example dataset: `random` gained 0.02019 with a 0.00519 optimism gap;
  `bayes` gained 0.02303 with a 0.00828 gap. It found the better configuration
  *and* flattered itself more, exactly as the ADR predicted. michi already
  printed both numbers side by side; that decision is now load-bearing rather
  than merely correct.

  Five constraints, all tested: the space stays printable and cannot be
  widened, nested scoring is not configurable, the optimism gap is always
  shown, the search is reproducible under a seed, and a missing dependency
  names the extra rather than silently falling back to a different sampler.

### Fixed
- Counting evaluated configurations assumed scikit-learn's
  `cv_results_["params"]`. Optuna's wrapper has no such key and records
  `trials_` instead, so the first Bayesian search to finish a fold raised
  `KeyError`. Found by running it, not by a test.

## [1.7.0] — 2026-07-27

The viewer finally shows what a terminal cannot draw.

### Added
- **Charts in `michi ui`.** The run detail page now draws four things from the
  manifest it already had: metrics with their confidence intervals, a
  confusion matrix, a calibration curve with its expected calibration error,
  and per-subgroup scores sorted worst-first.

  The data was recorded from v0.2 onward and simply never rendered — the
  viewer was 99 lines listing runs. A terminal can rank models and print a
  confusion table; it cannot draw a reliability diagram, which is the whole
  reason the viewer exists.

  Every chart is a rendering and computes nothing, so the viewer can never
  disagree with the terminal about the same run. They are inline SVG — no
  plotting library at render time, no CDN, no JavaScript — and a chart that
  cannot be drawn honestly is not drawn: more than ten classes shows the
  table instead.
- A stage map in `michi --help`, naming each verb by the kanji for what it
  does. `path` in the console draws the full table with the current context's
  readiness marked.

### Fixed
- **Charts were invisible in the viewer.** They shipped with a hardcoded
  near-black for text and rules, which is unreadable on the viewer's dark
  background. Text and rules now inherit the page's own colour, so the same
  chart is legible in the dark viewer and a light printed report. A test
  asserts it for every chart.
- Confusion-matrix shading is per row rather than per matrix. Shading by the
  whole matrix makes the majority class the only visible thing on an
  imbalanced problem, which hides exactly the failure the chart is read to
  find.
- Subgroup and metric labels no longer collide with their own bars.
- The CLI module docstring still said the console "will become" available in
  v0.5, two years of releases after it shipped.

## [1.6.0] — 2026-07-27

### Added
- **`michi ensemble`** — stack or soft-vote several models, and find out
  whether it was worth it. `--method stack` trains a meta-learner on
  out-of-fold member predictions; `--method vote` averages them.

  The ensemble is cross-validated **beside its own members** and tested with
  the same corrected resampled *t*-test, so the leaderboard answers the only
  question that matters. On the example dataset the stack scored 0.881 against
  linear's 0.884 and took 7.1s against 0.0s — a tie, reported as a tie. Every
  other tool prints the ensemble score alone, where a combination that gained
  nothing looks like a win.

  Each member carries its own preparation, because standardisation is right
  for a linear model and pointless for a tree; sharing one pipeline would
  impose one model's needs on all of them.

  michi picks no members, prunes none for scoring poorly, and weights none by
  validation score. Those are modelling judgements, and a tool that makes them
  silently is an AutoML system wearing a different hat.
- `register_transient` in the model registry: an ensemble is assembled from
  what the user named, so it joins the catalogue for the duration of one
  command and leaves cleanly — including when the run raises. Two tests pin
  that, because a leftover entry would make `bench --models ensemble` mean
  whatever the last ensemble command happened to build.

## [1.5.0] — 2026-07-27

Neural networks, and a scaling bug they exposed.

### Added
- **`mlp`** — a feed-forward neural network in the model catalogue, on
  scikit-learn's training loop. No extra install: the first network someone
  tries should not require a CUDA version and an afternoon.
- **`torch-mlp`** (`pip install 'komichi[torch]'`) — a real PyTorch training
  loop: epochs, Adam, mini-batches, a validation split, early stopping, and
  restoring the best weights rather than the last. This is the code an
  engineer retypes for the fortieth time, written once.

  Both are ordinary catalogue entries, so `bench`, `tune`, `fit`, and
  `predict` work on them with no special case anywhere in michi — a network
  is cross-validated, significance-tested, and held against the dummy baseline
  exactly like a random forest. michi never presents "we used deep learning"
  as a result.
- Search spaces for both, so `tune --model mlp` and `tune --model torch-mlp`
  work like any other model. Layer sizes, dropout, and learning rate are
  hyperparameters, not defaults michi picked for your data.

### Fixed
- **A recipe's fitted step smuggled raw magnitudes past the scaler.** When a
  recipe named a column, `transformer_specs` replaced michi's handling of it —
  including the standardisation scale-sensitive models depend on. An imputed
  salary reached the estimator in the tens of thousands while every other
  feature sat near zero.

  Found because the new `mlp` scored *exactly* the dummy baseline on a dataset
  where linear regression scored 0.89. It was never only about neural
  networks: `linear`, `ridge`, `lasso`, `knn`, and `svm` were all quietly
  degraded whenever a recipe had a fitted step. On the example dataset the
  maximum feature magnitude fell from 56,542 to 7.0, and `mlp` went from
  0.5000 to 0.8757. The recipe still decides *what* happens to a column;
  scaling is appended after it rather than instead of it.

## [1.4.0] — 2026-07-27

The last mile. Everything michi did stopped one step before the thing a
practitioner actually hands in.

### Added
- **`michi tune`** — hyperparameter search over a space you can print before
  anything runs (`--list-space`) and replace with your own YAML (`--space`).
  Random, successive-halving, and grid strategies, all from scikit-learn.

  The search is nested. Configurations are chosen by cross-validation *inside*
  each training fold, and the reported score comes from folds the search never
  touched. michi prints three numbers — the honest one, the same model
  untuned on the same folds, and the search's own optimistic best — because
  reporting an inner search score as performance is the second most common
  silent leak in tabular ML, after target encoding.
- **`michi fit`** — train one model on everything and save it. Reports no
  accuracy on purpose: a score measured on the rows a model trained on is the
  most confidently wrong number a tool can print.
- **`michi predict`** — predict on data with **no label column**, writing CSV,
  TSV, or parquet with an optional id column and class probabilities. This is
  what a competition submission, a batch scoring job, and a smoke test all
  need, and the one thing `eval` cannot do.

  It accepts anything `eval` does, including `module:object` — so a PyTorch,
  TensorFlow, or ONNX model works identically to a random forest, with no
  per-framework loader for michi to maintain and get wrong.
- `tune --save-params` writes YAML that `fit --params` reads, so the two verbs
  hand off without the user joining them by hand.

### Changed
- The roadmap's "known gaps" now distinguishes Bayesian search (needs an ADR;
  an optimiser proposes from its own beliefs) from deep learning (reachable
  today through the predict protocol and the plugin catalogue; what michi will
  not own is a training loop).

## [1.3.0] — 2026-07-27

Target encoding, and a menu for picking the columns to engineer.

### Added
- **`target-encode`** — replace a category with the target's mean for that
  category. The strongest encoding available for high-cardinality columns, and
  the easiest way to destroy a model: encoded naively, a unique id maps
  one-to-one onto its own label. On 600 random labels against 600 unique ids —
  data containing no signal whatsoever — the naive encoding cross-validates at
  **1.000** and michi's out-of-fold encoding at 0.517. A test asserts both
  halves of that comparison, because the trap is only instructive if you can
  see it spring.

  michi ships its own encoder rather than scikit-learn's. sklearn deprecated
  the parameter that makes `TargetEncoder` reproducible in 1.9 and removes it
  in 1.11, and the replacement needs a newer floor than michi's `>=1.4` —
  which would drop users to fix one operation. `export` writes the encoder
  class into the generated file, so exported pipelines still import nothing
  but pandas and scikit-learn, and the reader can see the arithmetic.
- **The feature menu.** `michi clean` with no operation flags now follows the
  findings triage with a column picker: operations grouped by what your
  columns make possible, nothing preselected, space to pick. `datepart` is
  offered when there are timestamps, `log` when something is actually skewed,
  `target-encode` when a categorical column is too wide for one-hot. michi
  lists shapes and stays quiet about worth — whether a product term helps your
  problem is domain knowledge michi does not have.

### Changed
- `examples/sweep.yaml` is now a verified plan rather than a hand-written one:
  12 cells in 6.1s against the example dataset, with the ranked output in
  `examples/README.md`. It was the last file in the repository claiming to be
  an example without having been run.
- The recipe file header listed seven operations and had not been updated
  since v1.0. It now lists all thirteen, split into cleaning and engineering,
  and the leakage note names every fitted step.

### Notes
- The scikit-learn floor stays at `>=1.4`.

## [1.2.0] — 2026-07-27

### Added
- **`bench --explain` now explains your own numbers**, not statistics in
  general. It states what the features bought over the dummy baseline, why
  the leader's interval is as wide as it is (the actual fold-to-fold spread,
  and how many rows each fold held out), why two tied models could not be
  separated (the measured gap against the measured noise), and roughly how
  many rows it would take to separate them. Every sentence is a fact about the
  result or arithmetic on it — a test scans the notes for advice-shaped
  phrasing, because "you should collect more data" is a judgement michi does
  not make.

## [1.1.0] — 2026-07-27

Feature engineering, and a path through the toolbox.

### Added
- **Five feature-engineering recipe operations.** `datepart` expands a
  timestamp into the components a model can actually use; `log` compresses a
  long right tail (`signed` for columns that genuinely go below zero);
  `interact` adds pairwise products or ratios; `binarize` reduces a column to
  above-a-threshold-or-not; `bin` discretises into quantile or uniform bins.
  Each is a recipe step like any other, so it inherits `apply`, `export`, and
  the fitted/deterministic split — `bin` learns its edges from data and is
  therefore fitted inside the cross-validation fold, while the other four run
  in `prepare()`.
- **`path`** — the stages of a tabular project, the command covering each, and
  a mark on the ones the current context could run right now. It executes
  nothing.
- **`walk`** — the stages one at a time, offering run / skip / stop at each.
  It asks and never suggests: no stage ranks the options or marks one
  recommended, the default is skip, and every stage prints the one-shot command
  it ran. ADR-0003 records the five constraints that keep a guided sequence
  from becoming workflow ownership; ADR-0001 stands.
- A console banner with a live inventory. The verb, model, operation, and
  explanation counts are read from the registries at startup rather than
  typed, so they cannot go stale — and none of those registries import pandas
  or scikit-learn, so the console still opens instantly.

### Fixed
- **`sweep` was unreachable from the console.** It was a verb of the CLI with
  no console entry, which is the flag-parity rule broken in the direction
  nobody checks.
- **The reproducing command printed by `clean` wrapped back to the left
  margin**, so a command too wide for the terminal read as two commands and
  got copied as one.
- A `ratio` interaction divided by zero into `pd.NA`, which forced the column
  to object dtype — precisely what a downstream estimator cannot consume.

### Notes
- **Target encoding is deliberately not included.** scikit-learn deprecated
  the parameter that makes `TargetEncoder` reproducible in 1.9 and removes it
  in 1.11; the replacement requires a newer scikit-learn than michi's floor.
  A step whose output changes between runs would break the reproducibility the
  rest of the toolbox rests on, so it waits for the floor to move.
- The recipe schema stays at 1.0. The new operations are additive vocabulary:
  a recipe written before them still loads, and an older michi meeting a newer
  recipe fails loudly by name rather than misreading it.

## [1.0.2] — 2026-07-27

### Fixed
- **A recipe with fitted steps produced an untrainable pipeline.** The
  transformer covered only the columns the recipe named and passed the rest
  through untouched, so a benchmark handed the estimator raw strings and every
  fold failed. A recipe now decides for the columns it speaks about, and
  michi's documented preparation covers the silence.
- **Generated code failed its own linter on wide recipes.** A six-column drop
  emitted a 128-character line; long literals are now wrapped exactly as the
  formatter would write them, and the tests cover a recipe wide enough to
  trigger it.
- A p-value that underflows to zero is reported as `p<0.0001` rather than
  `p=0`, which read as a bug rather than as the certainty it represents.

### Changed
- Every example in the README and in `examples/` is now real captured output
  from one run over one dataset, rather than illustrative text.

## [1.0.1] — 2026-07-27

Fixes found by using the built package the way a stranger would.

### Added
- **`michi.toml` now reaches every command**, not only the console. The
  documented precedence — flags > `michi.toml` > built-in — was previously
  true only inside the console, which made the documentation wrong for every
  shell user. `michi inspect`, `bench`, `clean`, and `report` now run with no
  arguments in a configured project.
- **`--recipe` on `eval`**, which the documentation had promised. Only the
  recipe's deterministic steps run: imputers in a recipe were fitted for
  training, and re-fitting them on the evaluation set would quietly change
  what is being measured.
- A PyPI publishing workflow using trusted publishing, which re-runs every
  gate against the tagged commit and refuses to publish if the tag and the
  package version disagree.

### Changed
- The distribution is **`komichi`** on PyPI, because `michi` was already
  taken. The command and the import are both still `michi`.
- A *configured* target is a hint, not a command: running against data that
  lacks it prints a note and continues, rather than failing a run the user did
  not misconfigure. A typed `--target` that is missing still fails loudly.

### Fixed
- Error messages naming an extra were rendered as terminal markup, so
  `pip install 'komichi[bench]'` reached users as `pip install 'komichi'` —
  an install command that silently does the wrong thing. Messages are now
  rendered verbatim, and a test asserts the brackets survive.
- The install command is built from one constant, and a test fails if any
  message hardcodes the package name — which immediately caught a stale hint
  in the Excel loader.
- **A skipped analysis is now a finding.** Above 200 columns michi caps its
  pairwise checks; it used to skip them in silence, so a user could conclude
  there were no duplicate columns when michi had never looked. Silence reads
  as "nothing found", which is a different claim from "not looked at".
- **`michi eval` was unusably slow on large data.** The default bootstrap on
  400,000 rows would have taken about three and a half minutes. Resamples are
  now shared across metrics instead of being redrawn identically for each,
  and very large draws are capped and rescaled — an 8× speed-up.

### Changed
- The distribution is **`komichi`** on PyPI (小道, *a small path*), because
  `michi` was taken. The command and the import remain `michi`.
- Large-sample intervals use the *m-out-of-n* bootstrap correction. Capping
  the draw without rescaling reported intervals nearly **three times too
  wide**; overstating uncertainty is no more honest than understating it.

## [1.0.0] — 2026-07-27

The public surface is frozen.

### Added
- **A quickstart** taking one messy CSV to a compared, reported model in
  fifteen minutes — the page that decides whether michi is worth adopting.
- **[ADR-0002](docs/adr/0002-freeze-the-public-surface.md)** records exactly
  what is now public API under semantic versioning: the profile, run manifest,
  and recipe schemas (all 1.0); the verb names, flags, and exit codes; the
  plugin entry-point groups and contract; and each package's `__all__`.
  Terminal layout, report styling, explanation prose, and finding thresholds
  are deliberately *not* frozen.
- **Schema-stability tests** that read committed 1.0 fixtures of all three
  artifact types on every CI run, execute every frozen recipe operation, and
  compile a 1.0 recipe to code. A promise nobody tests is a hope.

### Fixed
- A recipe step's `why` — the reason a column was dropped — was written as a
  YAML comment and silently lost on reload. It is now a field and round-trips.
  That reason is exactly the part nobody can reconstruct six months later, and
  the freeze tests are what caught it.

### Notes
- The freeze is what makes the toolbox claim real: a verb adopted today
  behaves the same next year, which is the only basis on which anyone puts
  michi in a pipeline.
- Everything on the roadmap has now shipped: `inspect`, `eval`, `bench`,
  `report`, `clean`/`apply`/`export`, the console, `sweep`, `ui`, and plugins.

## [0.8.0] — 2026-07-27

The plugin surface.

### Added
- **Two extension points**, discovered via entry points:
  `michi.models` adds algorithms to the `bench` catalogue, and
  `michi.adapters` adds ways to load a model for `eval` — which is where an
  ONNX or framework loader belongs, since michi deliberately ships none.
- **`michi plugins`** lists what is installed, working or not.
- **A published compatibility suite** (`michi.plugins.check_model_entry`,
  `check_adapter`) that plugin authors run in their own CI. michi's maintainer
  is not the integration point for every plugin — that is the only way an
  ecosystem stays affordable for one person to support.
- `CONTRIBUTING.md` and a plugin guide.

### Notes
- These points were opened at v0.8, not before, because the interfaces behind
  them had by then survived six milestones and a dozen concrete
  implementations without changing shape. An interface with one implementation
  is a guess, and a published interface is a promise.
- **A broken plugin is skipped, never fatal**: an entry point that fails to
  import is reported by `michi plugins` and ignored, and benchmarks keep
  running.
- **Built-ins win ties.** A plugin cannot shadow `rf` or any documented name,
  so `--list-models` always means what the documentation says.
- Recipe operations, report sections, and metric definitions remain closed.
  They stay closed until something real needs them.

## [0.7.0] — 2026-07-27

A local viewer: `michi ui`.

### Added
- **`michi ui`** (needs `michi[ui]`) — a read-only web view over the runs
  directory: every run grouped by dataset and target, and a detail page per
  run with metrics beside their baselines, checks, slices, and full
  provenance.
- Serves on **localhost only**, with no flag to change that.

### Notes
- The viewer is **read-only by construction**: a test asserts the application
  exposes only `GET`. A UI that can act is a platform, and michi is not one.
- **No database, no build step, no network.** Every request reads the
  directory, so a run appears on refresh; pages are server-rendered HTML with
  inline CSS, so the viewer works air-gapped and cannot rot when a frontend
  toolchain moves on.
- **It is deletable.** Removing it would remove convenience and not one
  capability — everything it shows is a file, and `michi report` renders the
  same artifacts. That is the bar it has to keep clearing.

## [0.6.0] — 2026-07-27

Experiment grids: `michi sweep`.

### Added
- **`michi sweep`** — executes a declarative grid of models × recipes × seeds
  from a YAML plan, recording one run manifest per cell. Progress is printed
  as it goes, and `--dry-run` lists the grid without running anything.
- **Content-hash caching and resumption.** Each cell's identity is a hash of
  the data, recipe, model, seed, and fold count that produce it, so an
  interrupted sweep resumes exactly where it stopped, and editing one recipe
  re-runs exactly the cells that use it. A long sweep *will* be interrupted;
  resumption is not a nicety.
- **Contained failure**: a cell that cannot train is recorded and the grid
  continues, because losing thirty completed cells to the thirty-first would
  be indefensible.
- **`--recipe` support in `bench`** — a recipe's deterministic steps run once
  up front; its fitted steps become a transformer fitted inside each fold, so
  comparing preprocessing strategies stays leak-free. A recipe the user wrote
  takes precedence over michi's default preparation.
- Sweep manifests feed straight into `michi report`.
- Documentation for `sweep`.

## [0.5.0] — 2026-07-27

The console: `michi` with no arguments.

### Added
- **An interactive console** with the context always visible in the prompt
  (`michi (train.csv → churned) ›`), `help`, `use`, `set`, `unset`, and
  `show [context|columns|models|runs]`.
- **Tab completion over your own data** — once a dataset is loaded, `set
  target <TAB>` completes that file's column names. This is the one thing a
  one-shot CLI can never do, and the reason the console exists.
- **`history --export`** writes the session as a replayable shell script of
  fully-expanded one-shot commands. Console-only commands never appear, so
  exploration stays reproducible.
- **`michi.toml`** — optional project defaults, found by searching upward from
  the working directory. Precedence is flags > `michi.toml` > built-in, and
  `michi info` prints which file was found and what it supplies. This is
  michi's answer to session state: if it is remembered, it is a file you can
  read, edit, diff, and commit.
- The console's context *is* the `michi.toml` model, so `save` persists it and
  a later session restores it.

### Fixed
- Console line splitting treated a backslash as an escape character, which
  silently mangled every Windows path a user typed. Backslashes are now
  literal; quoting still groups arguments containing spaces.

### Notes
- The console contains **zero logic**: every verb dispatches to the same Typer
  application the shell invokes, and a test asserts no console verb exists
  outside the CLI. Deleting the console would remove convenience, never
  capability.
- Bare `michi` prints help rather than opening a prompt when stdout is not a
  terminal, so scripts never hang.

## [0.4.0] — 2026-07-26

Cleaning decisions as a file you own: `michi clean`, `apply`, and `export`.

### Added
- **`michi clean`** — authors a **recipe**: an ordered list of declarative
  cleaning operations plus the schema they were written against. Your data is
  never touched. The interactive mode walks the *findings* `inspect` produced,
  grouped, so a two-hundred-column dataset yields a handful of questions
  rather than two hundred prompts — and "leave them as they are" is always
  present and always the default.
- **Full flag parity**: `--drop`, `--dedupe`, `--cast`, `--impute`, `--clip`,
  `--encode`, `--scale` express everything the wizard can, and every session
  prints the non-interactive command that reproduces it.
- **`michi apply`** — executes a recipe non-destructively to a new file. The
  recipe's schema snapshot acts as a data contract: applying it to data that
  lacks the named columns fails loudly, with `--no-strict` to degrade
  gracefully.
- **`michi export`** — compiles a recipe into a standalone Python module that
  imports pandas and scikit-learn, never michi. Deterministic steps become a
  `prepare()` function; fitted steps become an sklearn transformer to fit
  inside cross-validation. Generated code passes ruff check and format, and a
  test executes it and asserts it matches `michi apply`.
- **Recipe artifact** (schema 1.0) written as commented YAML through a
  template — `yaml.dump` cannot produce a comment, and the comments carry the
  reason each decision was made.
- Honest leakage reporting: steps that learn from data are marked as such,
  `apply` says when it fitted them on a whole file, and the caveat is written
  into the recipe itself.
- Documentation for `clean`, `apply`, and `export`.

## [0.3.0] — 2026-07-26

Honest model comparison: `michi bench` and `michi report`. This completes the
MVP — inspect, evaluate, compare, report.

### Added
- **`michi bench`** — cross-validates several models and reports which of them
  are actually distinguishable. A dummy baseline is always added; differences
  from the leader are tested with the **corrected resampled *t*-test**
  (Nadeau & Bengio, 2003) and Holm-adjusted across models, and the conclusion
  is stated in plain language ("tied with leader at this sample size").
- **A model catalogue** (`--list-models`) of 14 algorithms: `dummy`, `linear`,
  `ridge`, `lasso`, `tree`, `rf`, `extra-trees`, `hist-gbm`, `knn`, `svm`,
  `naive-bayes` in the base install, plus `xgb`, `lgbm`, `catboost` behind the
  `bench` extra. Everything trains locally; nothing is downloaded.
- **Stated column preparation**, printed every run and recorded in every
  manifest, and fitted *inside each fold* so a benchmark cannot leak.
  Overridable with `--impute`, `--encode`, `--no-scale`.
- **`michi report`** — renders recorded runs as a self-contained offline HTML
  page, Markdown for a pull request, or a paste-ready booktabs LaTeX table for
  a paper. Runs are grouped by dataset content hash and target, so
  incomparable numbers are never put in one table.
- Benchmark checks: `below-baseline`, `no-clear-winner`, `model-failed`.
- Documentation for `bench` and `report`.

### Changed
- Unknown or task-incompatible model names now fail immediately, before any
  training, rather than appearing as a failed row in the leaderboard.

### Fixed
- The corrected *t*-test returned "not significant" when two models differed
  by exactly the same amount on every fold. A zero-variance difference is the
  strongest possible evidence, not the weakest; the limit is now p = 0.

## [0.2.0] — 2026-07-26

Evaluate models michi never saw trained: `michi eval`.

### Added
- **`michi eval`** — evaluate an existing model against a dataset. Reports
  metrics with 95% bootstrap confidence intervals, always beside trivial
  baselines, plus a confusion matrix, per-slice scores, and a calibration
  curve.
- **Two ways to supply a model**: sklearn-compatible pickle/joblib files, or
  *any* object exposing `predict(X)` via a `mymodule:my_model` reference —
  which covers PyTorch, TensorFlow, ONNX, and bespoke models without michi
  shipping a loader per framework. Formats that cannot be loaded honestly are
  refused with a route forward rather than a confusing failure.
- **Eight evaluation checks**: `below-baseline`, `beats-baseline`,
  `suspiciously-perfect`, `single-class-predictions`, `wide-interval`,
  `miscalibrated`, `slice-gap`, and `small-evaluation-set` — each with an
  explanation and options, never a recommendation.
- **Run manifests** (schema 1.0) written to `runs/` by default: dataset and
  model hashes, seed, metrics, baselines, checks, and the environment,
  making any number traceable to the conditions that produced it.
- **CI gate**: `--fail-under metric=value`, respecting each metric's
  direction so `rmse=3.0` passes when RMSE is at most 3.0.
- Automatic task detection (classification or regression), overridable with
  `--task`.
- Documentation for `eval`, including the predict protocol.

### Changed
- sklearn's internal warnings are suppressed during evaluation; michi reports
  the same conditions itself, as checks, in language a user can act on.

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
