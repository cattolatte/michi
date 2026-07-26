# `michi clean` · `apply` · `export`

Cleaning decisions, captured as a file you own.

```bash
michi clean data.csv --target churned          # interactive triage
michi clean data.csv --drop id --impute age=median --cast amount=numeric
michi apply michi.recipe.yaml data.csv -o clean.parquet
michi export michi.recipe.yaml -o pipeline.py
```

## Why a recipe

An interactive session that changes your data and forgets what it did is
worthless the next day. So `michi clean` **never touches your data**. It
writes a *recipe*: an ordered list of declarative operations, plus the schema
they were written against.

That recipe is the product. It is reviewable in a pull request, versionable in
git, re-runnable on next month's export, and compilable into pipeline code.

```yaml
# michi recipe — cleaning decisions you made, as a file you own.
#
# Apply it:   michi apply michi.recipe.yaml data.csv -o clean.parquet
# Compile it: michi export michi.recipe.yaml -o pipeline.py

schema_version: "1.0"
target: churned

# The data this recipe was written against. Applying it to
# something with different columns fails loudly, on purpose.
source:
  sha256: "8b0f2ad7de6232b18ca83ec9d88c237d9018b82b46af93e7b7cb4891162eaa8a"
  n_rows: 120
  columns:
    age: numeric
    notes: empty

steps:
  # column was entirely missing
  - op: drop
    columns: ["notes"]
  - op: impute
    columns: ["salary"]
    strategy: median
```

Edit it by hand whenever you like — recipes are read with a plain YAML parser,
so a hand-written one works exactly like a generated one.

## `michi clean` — authoring

With no operation flags, `clean` profiles the data and walks you through the
**issues it found**, grouped:

```
  [2/6]  4 columns are some missing values
         salary 11.7% missing (14 of 120); age 3.2% missing (4 of 120)
         salary, age, fare, tenure

  What would you like to do?
  ❯ impute the median for 3 numeric column(s)
    impute the mean for 3 numeric column(s)
    impute the most frequent value for 1 column(s)
    drop the affected rows
    drop the columns
    leave them as they are
```

Two things matter here. It walks **findings, not columns** — a dataset with
two hundred columns produces a handful of grouped questions, not two hundred
prompts. And **"leave them as they are" is always present and always the
default**: michi lists the options and you choose.

When the session ends, michi prints the command that reproduces it:

```
  To reproduce this without the prompts:
    michi clean data.csv --drop notes,country --impute salary=median --target churned
```

Nothing in michi exists only interactively.

### Operations

| Operation | Flag | What it does |
|---|---|---|
| `drop` | `--drop a,b` | Remove columns |
| `dedupe` | `--dedupe` | Remove exactly duplicated rows |
| `cast` | `--cast col=numeric` | Parse as `numeric`, `datetime`, `category`, `string` |
| `impute` | `--impute col=median` | `median`, `mean`, `most_frequent`, `constant`, `drop_rows` |
| `clip` | `--clip col` | Bound values to the 1st–99th percentile |
| `encode` | `--encode col=onehot` | `onehot` or `ordinal` |
| `scale` | `--scale col=standard` | `standard`, `minmax`, `robust` |

Steps are ordered automatically — drop, dedupe, cast, impute, clip, encode,
scale — so each can assume the previous ran.

## `michi apply` — executing

```bash
michi apply michi.recipe.yaml data.csv -o clean.parquet
```

Non-destructive: the input is never modified, and nothing is written without
`--out`. Output format follows the extension (`.csv`, `.tsv`, `.parquet`).

The recipe's schema snapshot acts as a **data contract**. Applying it to data
that lacks the columns it names is an error, not a surprise:

```
error: data is missing 3 column(s) the recipe was written against:
country, notes, salary. Pass --no-strict to apply the steps that still
make sense.
```

### The leakage note

Imputation, encoding, and scaling *learn* from the data they see. When `apply`
fits them on a whole file, it says so:

```
note: impute fitted on all 120 rows of this file — for modelling, fit inside
the train/test split instead (michi export writes a pipeline that does)
```

That is the honest position. Fitting an imputer on your full dataset and
*then* splitting means your test fold influenced the training data. For
exploration it does not matter; for modelling it does, which is what `export`
is for.

## `michi export` — compiling

```bash
michi export michi.recipe.yaml -o pipeline.py
```

Compiles the recipe into a standalone Python module that imports pandas and
scikit-learn — **never michi**. A user who outgrows michi should leave with
working code, not a dependency.

The generated file has two pieces, because they carry different risks:

```python
def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the deterministic cleaning steps.

    Returns a new frame; the input is never modified.
    """
    frame = frame.copy()

    # column was entirely missing
    # Drop 1 column(s).
    frame = frame.drop(columns=["notes"], errors="ignore")
    ...


def build_pipeline() -> ColumnTransformer:
    """Build the transformer for steps that learn from data.

    Fit this inside your cross-validation, never on the whole dataset:
    an imputer fitted on all rows has already seen your test fold.
    """
    ...
```

`prepare` handles the deterministic steps — safe anywhere. `build_pipeline`
returns an sklearn transformer for the fitted steps, to compose into a
`Pipeline` and fit inside cross-validation.

Generated code passes `ruff check` and `ruff format --check`, and a test in
michi's own suite **executes it and asserts it produces the same result as
`michi apply`**. A code generator whose output is merely plausible is worse
than none.

## Options

### `clean`

| Option | Purpose |
|---|---|
| `--out`, `-o` | Where to write the recipe (default `michi.recipe.yaml`) |
| `--target`, `-t` | Label column |
| `--drop`, `--dedupe`, `--cast`, `--impute`, `--clip`, `--encode`, `--scale` | Operations, as above |
| `--no-input` | Never prompt; use only the flags given |
| `--sample` / `--full` / `--seed` | Sampling for large files |

### `apply`

| Option | Purpose |
|---|---|
| `--out`, `-o` | Where to write the transformed data |
| `--strict` / `--no-strict` | Fail, or degrade gracefully, when columns are missing |

### `export`

| Option | Purpose |
|---|---|
| `--out`, `-o` | Where to write the module (default: stdout) |
