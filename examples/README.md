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
| [`sweep.yaml`](sweep.yaml) | `michi sweep` | A grid of experiments: models × recipes × seeds — 12 cells, verified against this dataset |

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

michi sweep sweep.yaml
```

The sweep runs the twelve cells in about six seconds and ranks them:

```
  12 ran  ·  0 reused  ·  6.1s

  model    recipe   seed    score
  ─────────────────────────────────
  linear   recipe      1   0.8988
  rf       recipe      1   0.8967
  linear   recipe      2   0.8949
  linear   recipe      0    0.892
  …
  tree     recipe      1   0.8256
```

Note what the seeds show: `linear` ranges from 0.8907 to 0.8988 across four
seeds, and `rf` from 0.8779 to 0.8967. The gap between the best `linear` cell
and the best `rf` cell is smaller than the spread within either — which is the
whole reason `bench` refuses to declare a winner on this dataset.

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
