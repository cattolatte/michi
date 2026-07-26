# `michi inspect`

Profile a dataset and explain what stands out. It reads a file you already
have, writes nothing unless you ask, and never changes your data.

```bash
michi inspect data.csv
michi inspect data.csv --target churned --explain
michi inspect data.parquet --html profile.html --open
michi inspect data.csv --json profile.json --fail-on high
```

## What it reports

**Per column** — michi's own kind (numeric, categorical, boolean, datetime,
text, empty), missing counts, distinct values, distribution shape, and a
compact summary.

**Findings** — observations ranked by severity:

| Finding | Meaning |
|---|---|
| `empty-column` | Every value is missing |
| `constant-column` | A single distinct value throughout |
| `high-missing` / `missing` | Missing values, above or below the 50% mark |
| `duplicate-rows` | Exactly repeated rows |
| `duplicate-columns` | Columns holding identical values |
| `identifier-like` | A distinct value in every row |
| `high-cardinality` | Many distinct categories relative to row count |
| `high-skew` | A long-tailed numeric distribution |
| `outliers` | Values beyond 1.5×IQR |
| `highly-correlated` | Two columns carrying the same information |
| `numeric-stored-as-text` | Numbers behind separators or currency symbols |
| `datetime-stored-as-text` | Dates stored as strings |
| `mixed-types` | More than one Python type in a column |
| `class-imbalance` | A skewed target distribution (needs `--target`) |
| `leakage-suspect` | A feature that predicts the target almost perfectly (needs `--target`) |

Findings state what is true. What to do about each one is your decision —
`--explain` prints what a finding means and the options practitioners choose
between, without recommending any of them.

## Naming a target

```bash
michi inspect data.csv --target churned
```

Two checks only make sense once michi knows which column is the label: class
imbalance, and leakage. The leakage check flags a feature that predicts the
label almost perfectly — usually a value recorded *after* the outcome, which
will not exist when the model is actually used. michi raises the suspicion;
confirming it needs knowledge of how the data was produced.

## Large files

Files above 256 MB are randomly sampled to 200,000 rows so first contact stays
fast. Sampling is seeded and always recorded in the artifact and the report —
no number is ever silently based on part of the data.

```bash
michi inspect huge.parquet --sample 50000 --seed 7   # smaller, reproducible
michi inspect huge.parquet --full                    # read everything
```

## Output formats

| Flag | Result |
|---|---|
| *(none)* | Rich terminal report |
| `--explain` | Adds what each finding means and your options |
| `--html FILE` | One self-contained HTML file — no CDN, no JavaScript, opens offline |
| `--json FILE` | The profile artifact: versioned schema, safe to diff in CI |
| `--open` | Opens the HTML report in a browser |
| `--quiet` | Findings only, no per-column table |

## In CI

`--fail-on` turns michi into a data-quality gate:

```bash
michi inspect data/train.csv --target label --fail-on high --quiet
```

Exit codes: `0` nothing at or above the threshold, `1` something was found,
`2` the file could not be read.

## Options

| Option | Default | Purpose |
|---|---|---|
| `--target`, `-t` | none | Label column; enables imbalance and leakage checks |
| `--explain` / `--no-explain` | off | Meaning and options for each finding |
| `--html` | none | Write a self-contained HTML report |
| `--json` | none | Write the profile artifact |
| `--open` | off | Open the HTML report in a browser |
| `--sample` | 200000 | Rows kept when a large file is sampled |
| `--full` | off | Read every row regardless of size |
| `--seed` | 0 | Seed for reproducible sampling |
| `--max-columns` | all | Truncate the column table |
| `--quiet`, `-q` | off | Findings only |
| `--fail-on` | none | Exit non-zero at this severity (`high`, `warn`, `info`) |
