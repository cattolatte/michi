# `michi report`

Render the runs you have already recorded.

```bash
michi report                          # summarise runs/ in the terminal
michi report runs/ --out report.html --open
michi report runs/ --format markdown  # for a pull request
michi report runs/ --format latex     # for a paper
```

## What it reads

The manifests written by `michi eval` and `michi bench`. That is the whole
data model: **the runs directory is the database.** No index, no daemon, no
state that can drift from the files. A runs directory can be committed,
copied, or emailed and still make sense — and any of it can be read without
michi installed, because manifests are plain JSON.

## Grouping

Runs are grouped by **dataset content hash and target**, because those are
what make two numbers comparable at all. A run on `train.csv` and a run on
`train_v2.csv` never share a table, even if the files have similar names —
the hash decides, not the filename.

Within a group, runs are ranked by their headline metric, and any recorded
significance verdict travels with them. A benchmark run carries one;
a single `eval` has nothing to compare against, so it shows `—`.

## Formats

| Format | Use |
|---|---|
| terminal *(default)* | A quick look at what you have run |
| `--format html --out FILE` | A self-contained offline page: no CDN, no JavaScript |
| `--format markdown` | Paste into a pull request, an issue, or a README |
| `--format latex` | A booktabs table for a manuscript |

Every format renders the same artifacts, so a paper table and a browser page
can never disagree about which model won.

The LaTeX output is paste-ready — `booktabs`, escaped labels, and a caption
that states the test used:

```latex
% Requires \usepackage{booktabs}
\begin{table}[t]
\centering
\caption{Recorded runs on \texttt{churn.csv}, target \texttt{churned}.
Intervals are 95\%.}
\begin{tabular}{llrr}
\toprule
Run & Model & balanced accuracy & 95\% CI \\
\midrule
\texttt{2f0eeed3} & linear & 0.708 & 0.6278--0.7888 \\
...
```

## Robustness

One malformed manifest does not stop a report over ninety good ones —
unreadable files are skipped. An empty runs directory produces an actionable
message naming the commands that write manifests.

## Options

| Option | Default | Purpose |
|---|---|---|
| `source` | `runs` | Runs directory, or a single manifest file |
| `--out`, `-o` | stdout | Write the report to a file |
| `--format`, `-f` | `html` | `html`, `markdown`, or `latex` |
| `--open` | off | Open the written report in a browser |
