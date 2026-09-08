# Model summary scripts

Builds a coverage map of the OpenEval HuggingFace dataset: for each benchmark,
which models have responses and which items each one covers.

Requires `pyarrow`, `fsspec`, `requests`.

## 1. Build the summary

```bash
python build_model_summary.py --out model_summary
```

Writes into `--out`:

- `model_summary_full.json` — `{benchmark: {model: [item_id, ...]}}`
- `model_summary_counts.json` — item counts and coverage % per model
- `model_summary_counts.md` — a table: models, items, best-covered model,
  median coverage, coverage range
- `parts/<benchmark>.json` — per-benchmark cache (see the note below)

It reads only the `response_id` and `model.name` columns from the response
tables, so it touches a small part of the ~8.8 GB rather than downloading it.
A full run still takes a while; it retries on dropped connections.

**The cache never expires.** A benchmark with a file in `parts/` is skipped, so
a crashed run resumes cheaply — but a re-run after the dataset changes will
silently reuse the old data. **Delete `parts/` for a fresh build.**

## 2. Merge duplicate model names (optional)

Some models appear under more than one spelling, so they count as separate
models. This merges them and unions their item lists.

```bash
python normalize_model_names.py --parts model_summary/parts --out model_summary \
    --aggressive --tag normalized
```

- without `--aggressive`: merges case differences only (`Phi-4` / `phi-4`)
- with `--aggressive`: also merges punctuation differences
  (`Qwen1.5-0.5B-Chat` / `qwen-1.5-0.5b-chat`)

It writes the same three files under the `--tag` name, plus
`model_name_merges_<tag>.json` listing every merge and the full name map, so
any of it can be checked or undone.

Note that `--aggressive` compares names with all punctuation stripped, which is
blunt — it would also merge `gpt-4.1` with a hypothetical `gpt-41`. Every group
it formed on the current dataset was checked and correct, but the audit file is
there for a reason.

## Notes

- The response tables have no `item_id` column, so it is parsed out of
  `response_id` (`[item_id]_[model-slug]_[trial]`) and checked against the real
  item IDs. Anything that fails is counted and reported, never guessed.
- `summary_table.py` renders the markdown table and is imported by both scripts.
- The dataset is hardcoded as `Open-Eval-Commons/OpenEval` at the top of
  `build_model_summary.py`.
