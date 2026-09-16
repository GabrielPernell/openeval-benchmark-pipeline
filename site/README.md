# Homepage coverage explorer

Scripts that build the interactive coverage section on
[open-eval.com](https://open-eval.com) (repo `open-eval/open-eval.github.io`,
a single static `index.html` with no build step).

Pick a benchmark from a searchable dropdown and the page shows

```
moralbench · 88 archived items · 11 models · 968 responses

Download   [ Items  11 KB ]  [ Responses  51 KB ]   Browse all files ↗

Model          Items Covered   Coverage   Responses
deepseek-r1               88     100.0%          88
...
```

The site ships two files: `index.html` and `coverage.json`. The page holds the
markup, styling and rendering code; every number and download link comes from
the JSON. Refreshing the figures means regenerating the JSON, not editing the
page.

## Regenerating

```
python build_coverage_json.py          # writes site/coverage.json
python build_explorer.py --copy-json   # updates index.html, copies the JSON next to it
```

Both are safe to re-run: `build_explorer.py` replaces the blocks it wrote last
time rather than stacking new copies, and a second run is byte-identical.

`build_coverage_json.py` reads
`model_summary/model_summary_normalized_counts.json` (per-model items covered
and response counts, name variants merged) and `model_summary/parts/*.json`
(each benchmark's item total). Rebuild those with
`summaries/build_model_summary.py` first if the dataset has changed. Its one
network call lists the dataset's files and their sizes; `--tree FILE` reuses a
saved listing instead.

`build_explorer.py` also refreshes two sentences higher up in the section that
had gone stale (they said 24 benchmarks and ~70 models per benchmark). They
carry the real figures as plain text so they read correctly without JavaScript,
and are updated from the JSON once it loads.

## What's in coverage.json

About 97 KB for 28 benchmarks and 2,855 model rows.

```
as_of      the date shown under the table
dataset    repo, revision, and the URL prefix download links are built from
totals     benchmarks, distinct models, items, responses
benchmarks {name: {items, models, responses, rows, files}}
             rows  [model, items covered, responses]
             files {item: [[path, bytes]], response: [[path, bytes]]}
```

Coverage percentages are deliberately not stored. The page divides items
covered by the benchmark's item total, so a percentage can never disagree with
the two numbers printed beside it.

## Downloads

Links point at the benchmark's parquet files in the dataset, through
`/resolve/main/...?download=true`. A table that is one file gets a direct link;
a sharded one opens a list of its shards. Every link shows its size, because
they range from 11 KB (moralbench items) to 1.5 GB across 26 shards (cnndm
responses).

## Reading the JSON from somewhere else

The explorer fetches whatever `data-src` on `#cov-explorer` names, so pointing
the page at a copy hosted elsewhere is a one-attribute change, provided the
file has the shape above. The dataset's own `data_summary_*.json` does not: it
has per-model `n_items` and `coverage` but no response counts, no benchmark
item totals and no file listings, and its name is date-stamped. A benchmark
entry with no `files` still renders a working table, just with nothing to
download.

## Deploying

`index.html` and `coverage.json` have to go up together. A page deployed
without the JSON shows a short message and a link to the dataset instead of the
table; opened straight off disk as a `file://` page it does the same, because
browsers block a local page from fetching the file beside it. Serve the folder
over http to check it locally.

The site repo is `open-eval/open-eval.github.io`, and we have read access but
not write, so changes go up as a PR from a fork. The working clone lives at
`../../open-eval.github.io-main (1)/open-eval.github.io`, with `origin` on the
org repo and `fork` on ours:

```
python build_explorer.py --site "<clone>/index.html" --copy-json
cd "<clone>" && git add index.html coverage.json && git commit && git push fork <branch>
```

Because `build_explorer.py` rewrites everything between its markers, an edit
made directly in the clone is lost the next time it runs. When someone revises
the page there, fold the change back into this script — the way the intro
sentence was trimmed in both places.
