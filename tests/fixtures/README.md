# Test fixtures

Tiny datasets committed for fully-offline tests, including deliberately
pathological ones (mixed types, unicode, single row, all-null columns,
planted target leakage). Land from v0.1 with `inspect`.

Rules: small enough to review in a diff, never downloaded, never generated at
test time when determinism matters.
