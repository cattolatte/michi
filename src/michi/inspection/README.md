# michi.inspection

**Planned — v0.1. Not implemented yet; this stub is intentional.**

Dataset profiling behind the `michi inspect` verb: dtypes, missing values,
duplicates, cardinality, skew, imbalance, correlations, constant columns,
outliers, datetime detection — with sampling beyond a size threshold and
explanations attached to findings. Produces the **profile** artifact (JSON)
plus terminal and HTML renderings.

Named `inspection` rather than `inspect` to avoid shadowing the stdlib module;
the CLI verb is still `michi inspect`. See PLAN.md §9 and §15 (v0.1).
