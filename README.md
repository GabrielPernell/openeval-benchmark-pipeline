# OpenEval Conversion Pipeline

Turns raw benchmark data into the OpenEval item schema, and exports it in the
layout the [OpenEval HuggingFace dataset](https://huggingface.co/datasets/Open-Eval-Commons/OpenEval)
uses.

`item_schema.json` and `validator.py` are copied from
[open-eval/OpenEval](https://github.com/open-eval/OpenEval) without changes.
Keep them in sync with upstream.

## What's here

```
openeval_pipeline/
  convert_to_openeval.py     convert a benchmark -> JSON items (entry point)
  export_repo_parquet.py     convert -> the HF dataset's parquet layout (entry point)

  adapter_base.py            the interface each dataset loader implements
  schema_utils.py            builds one item dict
  repo_format.py             writes the HF dataset's parquet tables
  net_utils.py               HuggingFace fetching, with retries and pacing
  item_schema.json           |  copied from upstream,
  validator.py               |  do not edit

  loaders/                   one file per benchmark
    dsr_bench.py             all five subsets
    visionwebdev.py          all three task types (data is named Vision2Web)
    dochop.py                the single public release

  fetchers/                  getting data that isn't a simple table
    fetch_vision2web_cache.py   streams tar.gz archives, keeps only the text
    fetch_dochop_cache.py       reads metadata columns, skips the images

  summaries/                 model-coverage summaries of the HF dataset
    build_model_summary.py
    normalize_model_names.py
    summary_table.py

  data_cache/                fetched source data
  output/                    converted items, our JSON format
  repo_export/               converted items, the HF dataset's parquet format
```

## Converting a benchmark

```bash
python convert_to_openeval.py --dataset dsr_bench --subset main --mode hf \
    --out output/dsr_bench_main.json \
    --contributor-name "Your Name" --contributor-email "you@example.edu"
```

`--mode hf` fetches from HuggingFace; `--mode cache` reads a local file made by
one of the `fetchers/` scripts. The run prints four steps (load, map, write,
validate) and a violation report for anything that fails.

To check an existing file without converting again:

```bash
python convert_to_openeval.py --validate-only output/dsr_bench_main.json
```

On Windows, add `-X utf8` — some files are opened without an explicit encoding
and will otherwise use the system codepage.

## Exporting for the HF dataset

The dataset on HuggingFace stores things differently from `item_schema.json`:

- three separate tables — `bench/` (one row per benchmark), `item/`, `response/`
- `item_metadata.source` is a plain **string** that points at the `bench/` row,
  not the nested object the schema file shows
- `item_content.input` and `.references` are lists of **strings**, so per-item
  details (subset, category, and so on) are packed into a JSON string alongside
  the question — the same thing the existing HarmBench items do

`export_repo_parquet.py` writes that layout. It checks its table schema against
the live dataset before writing, so a mismatch fails loudly instead of producing
a file that won't merge.

```bash
python export_repo_parquet.py --out repo_export \
    --contributor-name "Your Name" --contributor-email "you@example.edu"
```

## The three benchmarks

| benchmark | items | source |
|---|---|---|
| DSR-Bench | 15,480 | five HF subsets that do **not** share a schema — different columns, and `code` uses a different split name and stores the question as a list |
| VisionWebDev | 193 | released as `zai-org/Vision2Web`. The parquet files on HF are only an index — truncated prompts, no ground truth. The real data (`workflow.json`) is inside 4.1 GB of tar.gz |
| DocHop | 2,074 | one 2.4 GB TSV. Each row holds a page image as base64 (~755 KB), so items reference the image instead of embedding it |

## Things worth knowing

**Every item fails validation on `responses`, and that's expected.** The
validator requires `responses` to be non-empty, but none of these three
benchmarks publishes model outputs — only the questions and answers. Filling in
`responses` means running models yourself or getting the results from the
authors. Everything else about the items validates cleanly.

**Empty `references` can be correct.** The schema marks `references` as
*required*, not *non-empty*, so an item with no reference answer still passes.
Sometimes that's right (open-ended safety questions are scored by a judge, not
against a gold answer) and sometimes it's a gap. The validator can't tell the
difference, so check per benchmark.

**Published metadata doesn't always match the data.** Vision2Web's index says
one number of test cases and its workflow files contain another, so the fetcher
counts from the real files and keeps the published figure separately.

**The summary scripts cache per benchmark and never expire it.**
`build_model_summary.py` skips any benchmark that already has a file in
`model_summary/parts/`, which makes a crashed run resumable but also means a
re-run after the dataset changes silently reuses old data. Delete
`model_summary/parts/` for a fresh build.

**The demo item's response wasn't produced by a model run.**
`output/dsr_bench_demo_item_with_response.json` exists to show the shape of a
complete item. Its answer was worked out by hand in an assistant session, and
the sampling settings are `null` rather than invented. Don't treat it as
evaluation data.

## Adding a benchmark

1. Write `loaders/<name>.py` with a `DatasetAdapter` subclass implementing
   `load_rows`, `row_to_content`, and `row_uid` (plus `repo_input` /
   `repo_references` if you want the parquet export).
2. Register it in `get_adapter()` in `convert_to_openeval.py`.
3. Run it.

Item assembly, ID generation, writing, and validation are all handled for you,
so a new benchmark is just "where do the rows come from, and how does one row
map to input/references/tags".

Before writing the loader, look at the real data first. All three benchmarks
here turned out differently from their documentation, and in two cases the
mapping had to be rewritten after checking actual files.

## Model summaries

`summaries/` builds a coverage map of the HF dataset — for each benchmark,
which models have responses and which items they cover.

```bash
python summaries/build_model_summary.py --out model_summary
python summaries/normalize_model_names.py --aggressive --tag normalized
```

The first reads only two columns from the response tables, so it touches a
small part of the ~8.8 GB rather than downloading it. The second merges models
recorded under more than one spelling (`Phi-4` and `phi-4`), writing an audit
file of every merge so it can be checked or undone.

Requires `pyarrow`, `fsspec`, and `requests`.
