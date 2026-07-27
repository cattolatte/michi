# Michi (道)

A local-first ML workbench: independent command-line tools that automate the
repetitive implementation work of machine learning while leaving every
judgement call to you.

**Automate implementation. Never automate judgement.**

## Start here

- [Quickstart](quickstart.md) — one messy CSV to a compared, reported model in
  fifteen minutes.
- [Philosophy](philosophy.md) — the principles every decision follows, and the
  permanent non-goals.
- [Roadmap](roadmap.md) — which verb ships when, and the artifacts they produce.
- [Architecture decisions](adr/README.md) — why the big calls were made.

## Install

```bash
pip install michi-ml
```

## First command

```bash
michi inspect data.csv --target label
```

Profiles the dataset, explains every finding, and writes nothing you did not
ask for. Add `--html profile.html` for a self-contained offline report, or
`--json profile.json` for a machine-readable artifact you can diff in CI.
