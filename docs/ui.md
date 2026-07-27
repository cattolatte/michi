# `michi ui`

Browse your recorded runs in a local, read-only web view.

```bash
pip install 'komichi[ui]'
michi ui              # serves http://127.0.0.1:8731 and opens a browser
michi ui runs/sweep --port 9000 --no-open
```

## What it is

A page listing every run in the directory, grouped by dataset and target, and
a detail page per run showing metrics beside their baselines, checks, slices,
and full provenance — the data hash, the model, the seed, and the library
versions that produced the numbers.

## What it deliberately is not

**It is read-only.** No route writes, deletes, or trains anything; a test
asserts that the application exposes only `GET`. A UI that can act is a
platform, and michi is not one.

**There is no database and no build step.** Every request reads the runs
directory, so a run appears on refresh. Pages are server-rendered HTML with
inline CSS — no JavaScript to bundle, nothing fetched from a CDN. The viewer
works on an air-gapped machine and cannot rot when a frontend toolchain moves
on.

**It binds to localhost only**, and the flag to change that does not exist.
michi does not serve anything to a network.

**It is deletable.** Removing the viewer would remove convenience and not a
single capability: everything it shows exists as a file, and `michi report`
renders the same artifacts. That is the bar the viewer has to keep clearing.

## Options

| Option | Default | Purpose |
|---|---|---|
| `runs_dir` | `runs` | Directory to view |
| `--port` | 8731 | Port on localhost |
| `--no-open` | off | Do not open a browser |

If the `ui` extra is not installed, michi says so and points at
`michi report`, which shows the same information as a file.
