# Examples

Real michi output, committed so you can read the product before installing it.

| File | Produced by | What it is |
|---|---|---|
| [`profile.json`](profile.json) | `michi inspect --json` | The profile artifact: versioned schema, every statistic and finding, safe to diff in CI |
| [`profile.html`](profile.html) | `michi inspect --html` | A self-contained report — 34 KB, no CDN, no JavaScript, opens offline |
| [`benchmark.html`](benchmark.html) | `michi bench --report` | A model comparison with confidence intervals and significance verdicts — 6 KB, also fully offline |
| [`recipe.yaml`](recipe.yaml) | `michi clean` | Cleaning decisions as a commented, hand-editable file |
| [`pipeline.py`](pipeline.py) | `michi export` | The recipe compiled into standalone Python — imports pandas and scikit-learn, never michi |
| [`sweep.yaml`](sweep.yaml) | hand-written | A grid of experiments: models × recipes × seeds |

```bash
michi inspect data/customers.csv --target purchased \
  --json profile.json --html profile.html

michi bench data/customers.csv --target churned \
  --models linear,rf,hist-gbm,knn --report benchmark.html

michi clean data/customers.csv --target purchased \
  --drop notes,country --cast amount_text=numeric \
  --impute salary=median --clip fare -o recipe.yaml
michi export recipe.yaml -o pipeline.py
```

The profiled dataset carries one instance of every problem michi detects — an
empty column, a constant column, a mostly-missing column, duplicated and
perfectly correlated columns, numbers stored as text, dates stored as text, an
imbalanced target, and a feature that leaks the label.

The benchmark shows the case michi exists for: several models score within a
point of each other, and the report says plainly that the data cannot
distinguish them rather than crowning a winner.

`pipeline.py` is the one worth opening. It is what a user leaves michi with:
a `prepare()` function for the deterministic steps and a `build_pipeline()`
factory for the ones that learn from data — separated precisely because they
carry different leakage risks, with the reason written into the file.
