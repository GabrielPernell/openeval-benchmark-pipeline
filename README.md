# OpenEval Conversion Pipeline

Converts raw benchmark rows into the OpenEval item schema (`item_schema.json`,
validated by `validator.py` -- both copied here from
[open-eval/OpenEval](https://github.com/open-eval/OpenEval), keep them in
sync with upstream if the schema changes).

Built while converting **DSR-Bench**, written so the same pipeline can be
pointed at **DocHop** and **VisionWebDev** once their data situations allow
it (see "Status of the other two datasets" below).

## How it's organized

```
openeval_pipeline/
  item_schema.json, validator.py   # copied verbatim from the OpenEval repo
  adapter_base.py                   # the interface every per-dataset loader implements
  schema_utils.py                    # shared item-assembly / id-generation / IO helpers
  convert_to_openeval.py              # CLI entry point
  loaders/
    dsr_bench.py                       # working adapter (this is the one actually run below)
    dochop.py                           # scaffold -- NotImplementedError, data not released yet
    visionwebdev.py                      # scaffold -- data released, loader not yet implemented/verified
  data_cache/
    dsr_bench_sample_raw.json             # 27 real DSR-Bench rows, fetched by hand for this sandbox demo
  output/                                  # conversion results land here
  demo_full_item_with_response.py           # one-off demo: what a fully valid item (incl. a real response) looks like
```

## Adding a new dataset

1. Write `loaders/<name>.py` with a class that subclasses `DatasetAdapter`
   (see `adapter_base.py`'s docstring) and implements `load_rows`,
   `row_to_content`, and `row_uid`.
2. Register it in `convert_to_openeval.py`'s `get_adapter()`.
3. Run it: `python3 convert_to_openeval.py --dataset <name> ...`

Everything else -- `item_id`/`response_id` generation, assembling the full
schema-shaped dict, writing JSON, running the validator and printing a
report -- is handled generically, so a new dataset is just "how do I get
rows, and how do I map one row to `{input, references, tags}`."

## Running it

```bash
python3 convert_to_openeval.py --dataset dsr_bench --subset main \
    --mode cache --source data_cache/dsr_bench_sample_raw.json \
    --out output/dsr_bench_main_items.json \
    --contributor-name "Your Name" --contributor-email "you@example.edu"
```

This prints a 4-step log (load rows -> map to items -> write JSON -> validate)
and a per-item violation report if anything fails schema validation.

To validate an existing file without re-converting:
```bash
python3 convert_to_openeval.py --validate-only output/dsr_bench_main_items.json
```

## Why network mode (`--mode hf`) won't work *in this sandbox*

This cloud session's outbound network proxy blocks `huggingface.co` (`curl`
returns `403 Forbidden` for both the main site and `datasets-server.huggingface.co`).
So `loaders/dsr_bench.py` was written with two paths:

- `mode="hf"` -- live fetch via HuggingFace's `datasets-server` REST API,
  paginating `https://datasets-server.huggingface.co/rows?...`. This is the
  code path a normal environment (your own machine, a CI runner, etc.)
  should use for full runs, since it doesn't require the `datasets` library
  or any auth for public datasets. **It will raise a clear `RuntimeError`
  in this sandbox** pointing at the network restriction.
- `mode="cache"` -- reads rows from a local JSON file. Used here to actually
  demo the pipeline end-to-end: I fetched 27 real rows spanning all 6
  DSR-Bench categories (associative/hashmap, associative/skip_list,
  hierarchical/heap, hierarchical/b_plus_tree, temporal/stack, network/graph)
  by hand via the same `datasets-server` REST API (through the WebFetch tool,
  which goes through a different, unrestricted path than this session's raw
  `curl`/`requests`), and saved them to `data_cache/dsr_bench_sample_raw.json`.

**Caveat on that cached sample:** WebFetch passes fetched content through a
summarization model rather than returning raw bytes, so it's not a fully
trustworthy way to pull exact data at scale -- I hit this directly (see
"Issues encountered" below). For a real, non-demo conversion run, use
`--mode hf` from an environment with normal network access, not the cached
sample.

## Field mapping (DSR-Bench -> OpenEval)

| OpenEval field | Source | Notes |
|---|---|---|
| `item_metadata.source.benchmark_name` | constant `"DSR-Bench"` | |
| `item_metadata.source.benchmark_version` | which subset (`main`/`challenge`/`spatial`/`natural`/`code`) | DSR-Bench has no single semantic version; the subset is the meaningful distinction |
| `item_metadata.source.paper_url` | `https://arxiv.org/abs/2505.24069` | |
| `item_metadata.source.dataset_url` | `https://huggingface.co/datasets/vitercik-lab/DSR-Bench-<subset>` | |
| `item_metadata.source.benchmark_tags` | `category`, `task`, `operation`, `length`, `prompt-style:<prompt>` | packs DSR-Bench's own taxonomy into the one field the schema offers for it |
| `item_content.input` | `[question]` | one self-contained prompt string per item (some rows embed a worked few-shot example inline -- see below) |
| `item_content.references` | `[ground_truth]` | kept as the exact string DSR-Bench stores it as (e.g. `"[7, 30, 11, 54, 69]"`), not re-parsed, to stay faithful to the release |
| `responses` | `[]` | see "About the `responses` field" |

## About the `responses` field (read this before assuming something's broken)

`validator.py` requires `responses` -- and each response's `scores` -- to be
**non-empty** lists (`_NON_EMPTY_LIST_KEYS = {'responses', 'scores'}`). Every
item this pipeline produces will fail validation on exactly that one field,
and that's expected: DSR-Bench (like DocHop and VisionWebDev) only released
the benchmark *items*, not any pre-computed model outputs. Per the schema
and `item_examples.json`, `responses[]` is meant to be filled in by whoever
actually runs a model on the item and scores it -- that's a separate step
from this conversion pipeline, which only handles `item_metadata` +
`item_content`.

Running the pipeline on the 27-row sample:
```
Validation summary: 0/27 items passed, 27 item(s) had violations.
```
...and every single violation is the same one: `responses [ValueError]`.
Nothing else -- confirming the `item_metadata`/`item_content` mapping itself
is schema-correct.

To show what a **fully** valid item looks like once a response exists,
`demo_full_item_with_response.py` takes item #0 (a short hashmap question),
has me (Claude, this session) actually solve it by hand, records that as a
genuine response with an exact-match score computed in code, and re-validates:
```
Validator result: PASS
```
See that script's docstring for an honesty note on `generation_parameters`
-- this harness doesn't expose literal sampling settings (temperature/top_k/
top_p) to me, so those are left `null` rather than invented (the schema's
`required` presence level explicitly permits `null`).

## Issues encountered (things worth knowing before trusting this blindly)

1. **This sandbox can't reach huggingface.co directly.** `curl`/`requests`
   from Bash get a `403` from the outbound proxy for both `huggingface.co`
   and `datasets-server.huggingface.co`. Worked around it using the WebFetch
   tool (which routes differently) to hand-fetch a 27-row sample. `--mode hf`
   in the pipeline is real, tested-shaped code, but untested end-to-end here
   for that reason -- worth a quick smoke test the first time it's run
   somewhere with normal network access.
2. **WebFetch isn't a reliable raw-data channel.** It summarizes/processes
   fetched content through a small model rather than always returning exact
   bytes -- I saw this directly: a 68-row request came back truncated
   mid-record, and independently, the `network`/`graph` (DFS) rows' `question`
   text contains a strange nested-bracket artifact in how the node list is
   printed (`[[[[[...51,,,,,,64,...]]]]]` instead of a plain list) that's
   suspiciously identical across different graphs whose edges reference node
   IDs not even present in that printed list. That pattern repeated
   identically across all 5 DFS rows I fetched, which suggests it's a
   genuine data-quality quirk in DSR-Bench's own released `question` text for
   that task (possibly worth flagging to the DSR-Bench authors separately),
   rather than something WebFetch invented -- but I can't fully rule out
   WebFetch mangling it further, since I only have its processed output to
   go on. Either way: **don't treat `data_cache/dsr_bench_sample_raw.json`
   as a byte-exact mirror of the source**, especially for the graph rows. A
   real run should use `--mode hf` (or the `datasets` library) with actual
   network access, which reads the raw parquet directly.
3. **`responses`/`scores` non-empty requirement** -- covered above, not a
   pipeline bug, just worth knowing up front rather than being surprised by
   a wall of validator output.

## Status of the other two datasets

- **DocHop**: still blocked on data release (see
  `claude/dataset-availability-check.md` in the project). `loaders/dochop.py`
  documents the intended field mapping (read from the paper) but
  `load_rows()` raises `NotImplementedError` with a pointer to email the
  corresponding author. Nothing to run yet.
- **VisionWebDev** (data hosted as `zai-org/Vision2Web`): data *is* released,
  but this session only checked the HuggingFace dataset-card description via
  WebFetch, not real files. `loaders/visionwebdev.py` has a draft mapping
  and a TODO list (clone the repo, inspect real task directories, confirm
  field names) before it's safe to trust. This is the natural next dataset
  to wire up.
