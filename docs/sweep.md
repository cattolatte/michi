# `michi sweep`

A grid of experiments, declared in a file.

```bash
michi sweep sweep.yaml
michi sweep sweep.yaml --dry-run     # list the grid, run nothing
michi sweep sweep.yaml --force       # ignore recorded results
```

## The plan

```yaml
# Models × recipes × seeds. Paths are relative to this file.
data: data/train.csv
target: churned
folds: 5

grid:
  models: [linear, rf, hist-gbm, xgb]
  recipes: [recipes/minimal.yaml, recipes/aggressive.yaml]
  seeds: [0, 1, 2]
```

That is 24 cells. `recipes` is optional; without it the grid is models × seeds.

A grid you can read, diff, and put in a paper's appendix is worth more than a
loop nobody else can run.

## Resumption is the point

A long sweep *will* be interrupted — machines sleep, sessions drop, someone
hits Ctrl-C. So re-running a sweep reuses what it already has:

```
    1/24  cached   linear · minimal · seed 0
    2/24  cached   linear · minimal · seed 1
    3/24  ran      rf · aggressive · seed 0
```

**Caching is by content, never by position.** Each cell's identity is a hash
of the data, the recipe, the model, the seed, and the fold count that produce
it. Edit one recipe and exactly the cells using it re-run; touch the data and
everything does; change nothing and nothing does.

```
$ michi sweep sweep.yaml            # 24 ran
$ michi sweep sweep.yaml            # 24 reused, 0.2s
$ vim recipes/minimal.yaml
$ michi sweep sweep.yaml            # 12 ran, 12 reused
```

## Failure is contained

One model that cannot train on one recipe is recorded as failed and the grid
continues. Losing thirty completed cells to the thirty-first would be
indefensible.

```
  22 ran  ·  0 reused  ·  2 failed  ·  184.3s
```

## Recipes in a sweep

A recipe's **deterministic** steps (drop, dedupe, cast, clip) run once, up
front. Its **fitted** steps (impute, encode, scale) become a transformer
fitted inside each fold — so a sweep comparing preprocessing strategies is
comparing them honestly, without any of them leaking.

When a recipe supplies fitted steps, they replace michi's default preparation:
a recipe you wrote takes precedence over michi's assumptions.

## Output

One run manifest per cell, written to `<runs_dir>/sweep/` by default, each
tagged with its cell, its key, and the whole plan — so a number is traceable
to the grid that produced it. They feed straight into reporting:

```bash
michi report runs/sweep --out results.html
michi report runs/sweep --format latex
```

## Options

| Option | Purpose |
|---|---|
| `--out`, `-o` | Where cell manifests are written |
| `--force` | Re-run every cell, ignoring recorded results |
| `--dry-run` | List the grid without running anything |
