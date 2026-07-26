# michi.evaluation

**Planned — v0.2. Not implemented yet; this stub is intentional.**

Rigorous evaluation of an *existing* model on a dataset behind the
`michi eval` verb: task-appropriate metrics, calibration, confusion,
per-slice metrics, an always-included dummy baseline, leakage checks, and
`--fail-under` CI gates. Writes the **run manifest** artifact.

Named `evaluation` rather than `eval` to avoid shadowing the built-in; the
CLI verb is still `michi eval`. See the project roadmap in the documentation.
