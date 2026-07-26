# Examples

Real michi output, committed so you can read the product before installing it.

| File | Produced by | What it is |
|---|---|---|
| [`profile.json`](profile.json) | `michi inspect --json` | The profile artifact: versioned schema, every statistic and finding, safe to diff in CI |
| [`profile.html`](profile.html) | `michi inspect --html` | A self-contained report — 34 KB, no CDN, no JavaScript, opens offline |
| [`benchmark.html`](benchmark.html) | `michi bench --report` | A model comparison with confidence intervals and significance verdicts — 6 KB, also fully offline |

```bash
michi inspect data/customers.csv --target purchased \
  --json profile.json --html profile.html

michi bench data/customers.csv --target churned \
  --models linear,rf,hist-gbm,knn --report benchmark.html
```

The profiled dataset carries one instance of every problem michi detects — an
empty column, a constant column, a mostly-missing column, duplicated and
perfectly correlated columns, numbers stored as text, dates stored as text, an
imbalanced target, and a feature that leaks the label.

The benchmark shows the case michi exists for: several models score within a
point of each other, and the report says plainly that the data cannot
distinguish them rather than crowning a winner.
