"""
response_format.py
===================
Builds `responses[]` entries, in both shapes the project needs:

  build_response_json()     nested, matching item_schema.json -- goes inside
                            output/<benchmark>_items.json
  build_response_parquet()  the HF dataset's response/ table encoding, which
                            differs: `generation_parameters` is a JSON *string*,
                            and `tools` / `external_resources` / `extra_artifacts`
                            / `scores` are structs of parallel arrays rather
                            than lists of objects.

RESPONSE_SCHEMA is copied from the live repo's response tables and is asserted
equal before writing, the same way repo_format.ITEM_SCHEMA is.
"""
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

_STR_LIST = pa.list_(pa.field("element", pa.string()))
_TYPE_CONTENT = pa.struct([("type", _STR_LIST), ("content", _STR_LIST)])

RESPONSE_SCHEMA = pa.schema([
    ("response_id", pa.string()),
    ("model", pa.struct([
        ("name", pa.string()),
        ("size", pa.string()),
        ("model_adaptation", pa.struct([
            ("system_instruction", pa.string()),
            # A JSON string in the repo's tables, not a nested struct.
            ("generation_parameters", pa.string()),
            ("tools", _TYPE_CONTENT),
        ])),
    ])),
    ("item_adaptation", pa.struct([
        ("request_input", _STR_LIST),
        ("demonstrations", _STR_LIST),
        ("external_resources", _TYPE_CONTENT),
    ])),
    ("response_content", _STR_LIST),
    ("scores", pa.struct([
        ("metric", pa.list_(pa.field("element", pa.struct([
            ("name", pa.string()),
            ("models", _STR_LIST),
            ("extra_artifacts", _TYPE_CONTENT),
        ])))),
        ("value", pa.list_(pa.field("element", pa.float64()))),
    ])),
])


def model_slug(name: str) -> str:
    """Repo ID convention: lowercase, underscores and spaces become hyphens."""
    return name.lower().replace("_", "-").replace(" ", "-")


def response_id(item_id: str, model_name: str, trial: int = 0) -> str:
    return f"{item_id}_{model_slug(model_name)}_{trial}"


def build_response_json(item_id, model_name, request_input, response_text,
                        scores, external_resources=(), generation_parameters=None,
                        system_instruction="", trial=0, size=None):
    """Nested form, for the JSON items this pipeline writes."""
    return {
        "response_id": response_id(item_id, model_name, trial),
        "model": {
            "name": model_name,
            "size": size,
            "model_adaptation": {
                "system_instruction": system_instruction,
                # Left as nulls when the source doesn't record them -- the
                # schema's `required` level permits null, and inventing
                # plausible sampling settings would be worse than saying
                # nothing.
                "generation_parameters": generation_parameters or {
                    "temperature": None, "do_sample": None, "top_k": None,
                    "top_p": None, "max_tokens": None,
                },
                "tools": [],
            },
        },
        "item_adaptation": {
            "request_input": list(request_input),
            "demonstrations": [],
            "external_resources": [dict(r) for r in external_resources],
        },
        "response_content": [response_text],
        "scores": [
            {
                "metric": {
                    "name": s["name"],
                    "models": list(s.get("models", [])),
                    "extra_artifacts": [dict(a) for a in s.get("extra_artifacts", [])],
                },
                "value": s["value"],
            }
            for s in scores
        ],
    }


def json_to_parquet_row(resp: dict) -> dict:
    """Convert the nested form into the repo's parallel-array encoding."""
    ma = resp["model"]["model_adaptation"]
    ext = resp["item_adaptation"]["external_resources"]
    return {
        "response_id": resp["response_id"],
        "model": {
            "name": resp["model"]["name"],
            "size": resp["model"]["size"],
            "model_adaptation": {
                "system_instruction": ma["system_instruction"],
                "generation_parameters": json.dumps(ma["generation_parameters"]),
                "tools": {"type": [t["type"] for t in ma["tools"]],
                          "content": [str(t["content"]) for t in ma["tools"]]},
            },
        },
        "item_adaptation": {
            "request_input": list(resp["item_adaptation"]["request_input"]),
            "demonstrations": list(resp["item_adaptation"]["demonstrations"]),
            "external_resources": {"type": [e["type"] for e in ext],
                                   "content": [str(e["content"]) for e in ext]},
        },
        "response_content": list(resp["response_content"]),
        "scores": {
            "metric": [
                {
                    "name": s["metric"]["name"],
                    "models": list(s["metric"]["models"]),
                    "extra_artifacts": {
                        "type": [a["type"] for a in s["metric"]["extra_artifacts"]],
                        "content": [str(a["content"]) for a in s["metric"]["extra_artifacts"]],
                    },
                }
                for s in resp["scores"]
            ],
            "value": [float(s["value"]) for s in resp["scores"]],
        },
    }


def write_responses(rows, out_dir, benchmark_slug, rows_per_shard=25000):
    os.makedirs(out_dir, exist_ok=True)
    shards = [rows[i:i + rows_per_shard] for i in range(0, len(rows), rows_per_shard)] or [[]]
    n = len(shards)
    paths = []
    for i, shard in enumerate(shards):
        table = pa.Table.from_pylist(shard, schema=RESPONSE_SCHEMA)
        assert table.schema.equals(RESPONSE_SCHEMA), "arrow schema drifted from the repo's"
        p = os.path.join(out_dir, f"{benchmark_slug}-{i:05d}-of-{n:05d}.parquet")
        pq.write_table(table, p)
        paths.append(p)
    return paths
