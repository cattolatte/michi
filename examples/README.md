# Examples

Real michi output, committed so you can read the product before installing it.

| File | What it is |
|---|---|
| [`profile.json`](profile.json) | The profile artifact written by `michi inspect --json`: versioned schema, every statistic and finding, safe to diff in CI |
| [`profile.html`](profile.html) | The self-contained report written by `michi inspect --html` — 34 KB, no CDN, no JavaScript, opens offline |

Both were produced from one deliberately messy dataset with:

```bash
michi inspect data/customers.csv --target purchased --json profile.json --html profile.html
```

The dataset carries one instance of every problem michi detects — an empty
column, a constant column, a mostly-missing column, duplicated and perfectly
correlated columns, numbers stored as text, dates stored as text, an
imbalanced target, and a feature that leaks the label.
