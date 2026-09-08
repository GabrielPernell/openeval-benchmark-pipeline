"""
repo_format.py
===============
Writes converted items in the layout the OpenEval HuggingFace dataset actually
uses (Open-Eval-Commons/OpenEval), which differs from the flat item_schema.json:

  bench/     one row per benchmark: name, version, paper_url, dataset_url, tags
  item/      <benchmark>-<shard>.parquet
  response/  <benchmark>-<shard>.parquet   (not written here -- we have none yet)

`item_metadata.source` is a bare STRING that acts as a foreign key into bench/,
and `item_content.input` / `.references` are `list<string>`. Per-item metadata
(subset, category, task, ...) has no field of its own, so the repo packs it into
a JSON object serialised into input[0] -- see the existing HarmBench items:

    {"prompt": "...", "category": "chemical_biological", "subset": "standard"}

Following that convention, each adapter supplies `repo_input(row)` (the dict to
serialise) and `repo_references(row)` (already-string references).

The arrow schema below is copied from the live repo's item tables; all 24 of
them share it exactly, and write_items() asserts a match before writing.
"""
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "v0.1.0"

CONTRIBUTOR_TYPE = pa.struct([
    ("name", pa.string()),
    ("email", pa.string()),
    ("affiliation", pa.string()),
])

ITEM_SCHEMA = pa.schema([
    ("item_id", pa.string()),
    ("item_metadata", pa.struct([
        ("ingestion_time", pa.string()),
        ("contributor", CONTRIBUTOR_TYPE),
        ("source", pa.string()),
    ])),
    ("item_content", pa.struct([
        # The repo names its list elements "element"; pa.list_(pa.string())
        # would name them "item" and break schema equality.
        ("input", pa.list_(pa.field("element", pa.string()))),
        ("references", pa.list_(pa.field("element", pa.string()))),
    ])),
    ("schema_version", pa.string()),
])

BENCH_SCHEMA = pa.schema([
    ("benchmark_name", pa.string()),
    ("benchmark_version", pa.string()),
    ("paper_url", pa.string()),
    ("dataset_url", pa.string()),
    # A real list in the repo, not a stringified one.
    ("benchmark_tags", pa.list_(pa.field("element", pa.string()))),
])


def _jsonify(value):
    """References are list<string>: pass strings through, serialise anything else."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def build_repo_item(row, adapter, idx, ingestion_time, contributor, item_id=None):
    """One raw row -> one repo-shaped item dict.

    `item_id` overrides the generated id. Pass it when the items already exist
    elsewhere (e.g. responses were built against them): a regenerated id would
    silently break the item<->response join.
    """
    payload = adapter.repo_input(row)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"repo_input() must return a non-empty dict (row {idx})")

    refs = [_jsonify(r) for r in adapter.repo_references(row)]

    return {
        # Repo convention: [lowercased benchmark name]_[timestamp]_[index].
        "item_id": item_id or f"{adapter.repo_source_name().lower()}_{ingestion_time}_{idx}",
        "item_metadata": {
            "ingestion_time": ingestion_time,
            "contributor": contributor,
            # Bare string FK into bench/ -- deliberately not the nested object
            # item_schema.json describes.
            "source": adapter.repo_source_name(),
        },
        "item_content": {
            "input": [json.dumps(payload, ensure_ascii=False)],
            "references": refs,
        },
        "schema_version": SCHEMA_VERSION,
    }


def write_items(items, out_dir, benchmark_slug, rows_per_shard=50000):
    """Write item/<slug>-<i>-of-<n>.parquet with the repo's exact arrow schema."""
    os.makedirs(out_dir, exist_ok=True)
    shards = [items[i:i + rows_per_shard] for i in range(0, len(items), rows_per_shard)] or [[]]
    n = len(shards)
    paths = []
    for i, shard in enumerate(shards):
        table = pa.Table.from_pylist(shard, schema=ITEM_SCHEMA)
        assert table.schema.equals(ITEM_SCHEMA), "arrow schema drifted from the repo's"
        p = os.path.join(out_dir, f"{benchmark_slug}-{i:05d}-of-{n:05d}.parquet")
        pq.write_table(table, p)
        paths.append(p)
    return paths


def write_bench(bench_rows, out_path):
    """Write the bench/ registry rows for the benchmarks we are contributing."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = [{**r, "benchmark_tags": list(r["benchmark_tags"])} for r in bench_rows]
    table = pa.Table.from_pylist(rows, schema=BENCH_SCHEMA)
    assert table.schema.equals(BENCH_SCHEMA), "bench schema drifted from the repo's"
    pq.write_table(table, out_path)
    return out_path
