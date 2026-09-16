#!/usr/bin/env python3
"""
build_coverage_json.py
======================
Build `coverage.json`, the data file the open-eval.com coverage explorer reads.

The explorer used to have all 28 benchmark tables written straight into
index.html. This replaces that: the page ships one small JSON file and renders
the tables from it, so refreshing the numbers means regenerating this file
instead of regenerating the page.

What goes in it
---------------
Per benchmark:
  items      total items in the benchmark's item table (coverage denominator)
  models     how many models have at least one response
  responses  total response rows
  rows       [model, items covered, responses] per model
  files      the benchmark's parquet shards on HuggingFace, with byte sizes,
             which is what the per-benchmark download links point at

Coverage percentages are deliberately NOT stored. The page divides items
covered by the benchmark total, so a percentage can never disagree with the two
numbers shown beside it.

Sources
-------
model_summary/model_summary_normalized_counts.json
    Per-model items covered and response counts, name variants already merged.
model_summary/parts/*.json
    Each benchmark's item total (the coverage denominator).
HuggingFace tree API
    File paths and sizes for the download links. This is the only network call;
    pass --tree to reuse a saved listing instead.

Usage
-----
    python build_coverage_json.py
    python build_coverage_json.py --out ../../open-eval.github.io-main/coverage.json
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

import requests

REPO = "Open-Eval-Commons/OpenEval"
TREE_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true&expand=true"
DATASET_URL = f"https://huggingface.co/datasets/{REPO}"
RESOLVE = f"{DATASET_URL}/resolve/main/"

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
COUNTS = os.path.join(PIPE, "model_summary", "model_summary_normalized_counts.json")
PARTS = os.path.join(PIPE, "model_summary", "parts")
DEFAULT_OUT = os.path.join(PIPE, "site", "coverage.json")

SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.parquet$")
NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def bench_key(path: str) -> str:
    """`item/anthropic_red_teaming-00000-of-00001.parquet` -> `anthropic-red-teaming`.

    Filenames on the repo use underscores; the summary and the page use
    hyphens. Normalizing here keeps one spelling everywhere downstream.
    """
    return SHARD_RE.sub("", path.split("/", 1)[1]).replace("_", "-")


def fetch_tree() -> list:
    """Every file in the dataset repo. The tree API pages at 50 entries."""
    url, out, pages = TREE_API, [], 0
    while url:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        out += r.json()
        pages += 1
        m = NEXT_RE.search(r.headers.get("Link", ""))
        url = m.group(1) if m else None
    print(f"  tree: {pages} pages, {len(out)} entries", file=sys.stderr)
    return out


def files_by_bench(tree: list) -> dict:
    """{benchmark: {'item': [[path, size], ...], 'response': [...]}}, shard-ordered."""
    out = {}
    for entry in tree:
        path = entry.get("path", "")
        if entry.get("type") != "file" or not path.endswith(".parquet"):
            continue
        table = path.split("/", 1)[0]
        if table not in ("item", "response"):
            continue          # bench/ is the registry, submissions/ isn't merged yet
        size = entry.get("size")
        if size is None:
            raise RuntimeError(f"no size for {path}")
        out.setdefault(bench_key(path), {}).setdefault(table, []).append([path, size])
    for tables in out.values():
        for shards in tables.values():
            shards.sort()
    return out


def load_summary():
    with open(COUNTS, encoding="utf-8") as f:
        counts = json.load(f)
    totals = {}
    for p in sorted(glob.glob(os.path.join(PARTS, "*.json"))):
        with open(p, encoding="utf-8") as f:
            part = json.load(f)
        totals[part["key"]] = part["n_items"]
    missing = sorted(set(counts) - set(totals))
    if missing:
        raise RuntimeError(f"no item total for {missing}; rebuild model_summary/parts")
    return counts, totals


def build(counts, totals, files, as_of):
    benchmarks = {}
    for name in sorted(counts):
        models = counts[name]
        if name not in files:
            raise RuntimeError(f"{name} is in the summary but has no parquet files on {REPO}")
        for table in ("item", "response"):
            if not files[name].get(table):
                raise RuntimeError(f"{name} has no {table}/ shards on {REPO}")
        # Highest coverage first, so the page can render rows in file order.
        rows = sorted(models.items(),
                      key=lambda kv: (-kv[1]["n_items"], -kv[1]["responses"], kv[0]))
        benchmarks[name] = {
            "items": totals[name],
            "models": len(models),
            "responses": sum(m["responses"] for m in models.values()),
            "rows": [[m, v["n_items"], v["responses"]] for m, v in rows],
            "files": {"item": files[name]["item"], "response": files[name]["response"]},
        }

    return {
        "as_of": as_of,
        "generated": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {"repo": REPO, "revision": "main",
                    "url": DATASET_URL, "resolve": RESOLVE},
        "totals": {
            "benchmarks": len(benchmarks),
            "models": len({m for b in counts.values() for m in b}),
            "items": sum(b["items"] for b in benchmarks.values()),
            "responses": sum(b["responses"] for b in benchmarks.values()),
        },
        "row_fields": ["model", "items_covered", "responses"],
        "benchmarks": benchmarks,
    }


def as_of_today():
    fmt = "%#d %b %Y" if os.name == "nt" else "%-d %b %Y"
    return datetime.date.today().strftime(fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write coverage.json")
    ap.add_argument("--tree", help="a saved tree listing to use instead of the HF API")
    ap.add_argument("--as-of", default=None, help="date shown on the page (default: today)")
    args = ap.parse_args()

    if args.tree:
        with open(args.tree, encoding="utf-8") as f:
            tree = json.load(f)
    else:
        tree = fetch_tree()

    counts, totals = load_summary()
    data = build(counts, totals, files_by_bench(tree), args.as_of or as_of_today())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=False)
        f.write("\n")

    t = data["totals"]
    n_rows = sum(len(b["rows"]) for b in data["benchmarks"].values())
    n_files = sum(len(b["files"][k])
                  for b in data["benchmarks"].values() for k in ("item", "response"))
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")
    print(f"  {t['benchmarks']} benchmarks, {t['models']} distinct models, "
          f"{n_rows} model rows")
    print(f"  {t['items']:,} items, {t['responses']:,} responses, "
          f"{n_files} downloadable files")
    print(f"  as of {data['as_of']}")


if __name__ == "__main__":
    main()
