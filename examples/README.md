# Examples

Real michi output, committed so you can read the product before installing it.
Every file here comes from one run over one deliberately messy dataset —
600 rows, 14 columns, 14 findings.

| File | Produced by | What it is |
|---|---|---|
| [`profile.json`](profile.json) | `michi inspect --json` | The profile artifact: versioned schema, every statistic and finding, safe to diff in CI |
| [`profile.html`](profile.html) | `michi inspect --html` | A self-contained report — 32 KB, no CDN, no JavaScript, opens offline |
| [`recipe.yaml`](recipe.yaml) | `michi clean` | Cleaning decisions as a commented, hand-editable file |
| [`pipeline.py`](pipeline.py) | `michi export` | The recipe compiled into standalone Python — imports pandas and scikit-learn, never michi |
| [`benchmark.html`](benchmark.html) | `michi bench --report` | A model comparison with confidence intervals and significance verdicts — 6 KB, also fully offline |
| [`sweep.yaml`](sweep.yaml) | hand-written | A grid of experiments: models × recipes × seeds |

```bash
michi inspect data/customers.csv --target purchased \
  --json profile.json --html profile.html

michi clean data/customers.csv --target purchased \
  --drop notes,country,country_copy,record_id,outcome_code,age_months \
  --cast amount_text=numeric --cast signup_date=datetime \
  --impute salary=median --clip fare -o recipe.yaml

michi export recipe.yaml -o pipeline.py

michi bench data/customers.csv --target purchased --recipe recipe.yaml \
  --models linear,rf,hist-gbm,knn --report benchmark.html
```

The dataset carries one instance of every problem michi detects — an empty
column, a constant column, a mostly-missing column, duplicated and perfectly
correlated columns, numbers stored as text, dates stored as text, an
imbalanced target, and a feature that leaks the label.

Two files are worth opening. **`benchmark.html`** shows the case michi exists
for: three models score within two points of each other, and the report says
plainly that the data cannot separate them rather than crowning a winner.
**`pipeline.py`** is what you leave with — a `prepare()` function for the
deterministic steps and a `build_pipeline()` factory for the ones that learn
from data, separated precisely because they carry different leakage risks,
with the reason written into the file.
