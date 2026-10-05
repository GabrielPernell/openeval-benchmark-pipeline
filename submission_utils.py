"""
submission_utils.py
===================
Shared by the scripts that write a submission folder for the OpenEval HF
dataset: gzipped item-schema JSONL, one item per line with its responses
nested inside, as the dataset's submissions/ PRs expect.
"""
import collections
import gzip
import json

_MISSING_VALIDATOR = """validator.py is missing. It belongs to open-eval/OpenEval rather than this
repo, so it is not checked in here. Copy validator.py and item_schema.json
from https://github.com/open-eval/OpenEval into the pipeline root, then run
this again."""

try:
    import validator
except ModuleNotFoundError:
    raise SystemExit(_MISSING_VALIDATOR)


def check_items(items):
    """Everything a submission must satisfy. -> list of problem strings."""
    problems = []
    item_ids = [it["item_id"] for it in items]
    if len(item_ids) != len(set(item_ids)):
        problems.append(f"{len(item_ids) - len(set(item_ids))} duplicate item ids")
    ids = [r["response_id"] for it in items for r in it["responses"]]
    if len(ids) != len(set(ids)):
        problems.append(f"{len(ids) - len(set(ids))} duplicate response ids")
    empty = [it["item_id"] for it in items if not it["responses"]]
    if empty:
        problems.append(f"{len(empty)} items have no responses: {empty[:5]}")
    vio = collections.Counter()
    for it in items:
        ok, vios = validator.validate_entry(it)
        for v in vios:
            vio[f"{v['field']} ({v['violation_type'].__name__})"] += 1
    problems += [f"validator: {k} x{n}" for k, n in sorted(vio.items())]
    return problems


def write_jsonl_gz(items, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def counts(items):
    """-> (items, responses, distinct models)"""
    models = {r["model"]["name"] for it in items for r in it["responses"]}
    return len(items), sum(len(it["responses"]) for it in items), len(models)
