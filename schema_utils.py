"""
schema_utils.py
================
Shared helpers for converting raw benchmark rows into the OpenEval item schema
(see item_schema.json / https://github.com/open-eval/OpenEval).

These helpers are dataset-agnostic on purpose: `loaders/*.py` files supply the
dataset-specific pieces (how to fetch rows, how to map a row to item_content),
and this module assembles the actual OpenEval item dict + writes/validates output.
"""
import json
import os
from datetime import datetime, timezone

SCHEMA_VERSION = "v0.1.0"


def generate_time() -> str:
    """ISO-8601-ish UTC timestamp in the same format used by item_examples.json
    (e.g. '20260220T224823Z')."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generate_item_id(benchmark_name: str, ingestion_time: str, idx: int) -> str:
    slug = benchmark_name.replace("_", "-").replace(" ", "-").lower()
    return f"{slug}_{ingestion_time}_{idx}"


def generate_response_id(item_id: str, model_name: str, idx: int) -> str:
    slug = model_name.replace("_", "-").replace(" ", "-").lower()
    return f"{item_id}_{slug}_{idx}"


def build_item(
    row: dict,
    adapter: "DatasetAdapter",
    idx: int,
    ingestion_time: str,
    contributor: dict,
    responses: list | None = None,
) -> dict:
    """
    Assemble one full OpenEval item dict from a single raw benchmark row.

    Args:
        row: one raw row as returned by adapter.load_rows(...)
        adapter: a DatasetAdapter (see adapter_base.py) describing this benchmark
        idx: 0-based index of this row within the current conversion run (used for item_id)
        ingestion_time: shared timestamp for the whole run (item_metadata.ingestion_time)
        contributor: {"name": ..., "email": ..., "affiliation": ...}
        responses: optional list of already-built response dicts (schema: responses[]).
                    Left as [] by default -- see README.md "About the `responses` field"
                    for why this is usually empty right after conversion.

    Returns:
        A dict matching item_schema.json.
    """
    item_id = generate_item_id(adapter.benchmark_name, ingestion_time, idx)
    content = adapter.row_to_content(row)  # {"input": [...], "references": [...], "tags": [...]}

    item = {
        "item_id": item_id,
        "item_metadata": {
            "ingestion_time": ingestion_time,
            "contributor": contributor,
            "source": {
                "benchmark_name": adapter.benchmark_name,
                "benchmark_version": adapter.benchmark_version,
                "paper_url": adapter.paper_url,
                "dataset_url": adapter.dataset_url,
                "benchmark_tags": adapter.base_tags + content.get("tags", []),
            },
        },
        "item_content": {
            "input": content["input"],
            "references": content["references"],
        },
        "responses": responses if responses is not None else [],
        "schema_version": SCHEMA_VERSION,
    }
    return item


def save_json(items: list, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)
