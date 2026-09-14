#!/usr/bin/env python3
"""
build_model_summary.py
=======================
Build a model-coverage summary for the OpenEval HF dataset: for every
benchmark, which models have responses, which items each one covers, and how
many responses each one contributed.

Coverage is NOT a complete grid -- models are added in waves and can be missing
from a benchmark or have gaps inside one -- so "model X was evaluated on
benchmark Y" does not imply it covers all of Y's items. That is what this
records.

Items covered and responses are different numbers. A model run several times on
the same item (trials 0, 1, 2 at the end of the response_id) covers that item
once but contributes several responses, so both are recorded.

Artifacts written:
  <out>/model_summary_full.json    {benchmark: {model: [item_id, ...]}}
  <out>/model_summary_counts.json  {benchmark: {model: {n_items, coverage, responses}}}
  <out>/model_summary_counts.md    the same counts as a README-ready table

Method
------
The response tables carry no item_id column, so it is recovered from
response_id, which follows `[item_id]_[model-slug]_[trial]`. Every recovered id
is checked against the item table's real ids; anything that fails to parse is
counted and reported rather than silently becoming a fake id. Such a response
still counts toward its model's responses -- it exists, it just can't be tied
to an item.

Only the `response_id` and `model.name` leaf columns are read. Parquet is
columnar and HuggingFace serves range requests, so this touches a small
fraction of the ~8.8 GB of response data rather than downloading it.
"""
import argparse
import collections
import json
import os
import re
import sys

import fsspec
import pyarrow.parquet as pq
import requests

from summary_table import render_table

REPO = "Open-Eval-Commons/OpenEval"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.parquet$")

# Bump whenever the part file format changes. A part written in an older format
# is rebuilt rather than reused, so adding a field can't leave stale cached
# parts silently missing it. v2 added per-model response counts.
PART_VERSION = 2


def model_slug(name: str) -> str:
    """Repo ID convention: lowercase, underscores and spaces become hyphens."""
    return name.lower().replace("_", "-").replace(" ", "-")


def with_retry(fn, what, attempts=5):
    """HF range reads occasionally return a truncated payload; retry those.

    A whole-run crash after an hour of streaming is not an acceptable failure
    mode, so transient network errors are retried with backoff.
    """
    import time
    delay = 3.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"      ! {type(e).__name__} on {what}; retry {i+1}/{attempts-1} "
                  f"in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def part_version(path: str):
    """Read a part file's format version without parsing the whole file.

    Parts can run to tens of MB, and "version" is written as the first key, so
    the first few bytes are enough.
    """
    with open(path, encoding="utf-8") as f:
        m = re.match(r'\{"version":\s*(\d+)', f.read(64))
    return int(m.group(1)) if m else 1


def list_tables(kind: str) -> dict:
    tree = requests.get(API + kind, timeout=120, headers=HEADERS).json()
    out = collections.defaultdict(list)
    for e in tree:
        name = e["path"].split("/")[-1]
        out[SHARD_RE.sub("", name)].append(e["path"])
    return {k: sorted(v) for k, v in out.items()}


def item_ids_for(paths, fs) -> set:
    ids = set()
    for p in paths:
        ids.update(pq.read_table(fs.open(BASE + p, "rb"), columns=["item_id"])
                     .column("item_id").to_pylist())
    return ids


def item_id_from(response_id: str, name: str, known: set):
    """Strip `_<trial>` then `_<model-slug>`; verify against the real item ids."""
    head = response_id.rsplit("_", 1)[0]           # drop trial index
    suffix = "_" + model_slug(name)
    if head.endswith(suffix):
        candidate = head[: -len(suffix)]
        if candidate in known:
            return candidate
    # Fall back to a longest-prefix search for ids that predate the convention.
    parts = response_id.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in known:
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_summary")
    ap.add_argument("--only", default=None, help="one benchmark stem, for testing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fs = fsspec.filesystem("http")

    items = list_tables("item")
    responses = list_tables("response")
    stems = sorted(set(items) & set(responses))
    if args.only:
        stems = [s for s in stems if s == args.only]
    print(f"{len(stems)} benchmarks to summarise", file=sys.stderr)

    parts_dir = os.path.join(args.out, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    # One file per benchmark so a failure costs only the current benchmark and
    # a re-run skips what is already done.
    for stem in stems:
        part = os.path.join(parts_dir, f"{stem}.json")
        if os.path.exists(part):
            cached = part_version(part)
            if cached == PART_VERSION:
                print(f"  {stem:<24} (already done, skipping)", file=sys.stderr)
                continue
            print(f"  {stem:<24} (cached part is format v{cached}, rebuilding)",
                  file=sys.stderr)
        known = with_retry(lambda: item_ids_for(items[stem], fs), f"{stem} items")
        key = next(iter(known)).rsplit("_", 2)[0] if known else stem
        by_model = collections.defaultdict(set)
        responses_by_model = collections.Counter()
        unparsed = 0
        for p in responses[stem]:
            def read_shard(p=p):
                out = []
                pf = pq.ParquetFile(fs.open(BASE + p, "rb"))
                for batch in pf.iter_batches(batch_size=50000,
                                             columns=["response_id", "model.name"]):
                    d = batch.to_pydict()
                    if "model" in d:
                        names = [(m or {}).get("name") for m in d["model"]]
                    else:
                        names = d["name"]
                    out.append((d["response_id"], names))
                return out
            for rids, names in with_retry(read_shard, p.split("/")[-1]):
                for rid, mname in zip(rids, names):
                    # Every row is a response from this model, whether or not
                    # its item_id can be recovered.
                    responses_by_model[mname] += 1
                    iid = item_id_from(rid, mname or "", known)
                    if iid is None:
                        unparsed += 1
                        continue
                    by_model[mname].add(iid)
        models_here = sorted(set(by_model) | set(responses_by_model))
        with open(part, "w", encoding="utf-8") as f:
            # "version" must stay the first key -- part_version() reads it from
            # the start of the file.
            json.dump({"version": PART_VERSION, "key": key, "n_items": len(known),
                       "unparsed": unparsed,
                       "responses": {m: responses_by_model[m] for m in models_here},
                       "models": {m: sorted(by_model.get(m, ())) for m in models_here}}, f)
        print(f"  {key:<24} items={len(known):<7} models={len(models_here):<4} "
              f"responses={sum(responses_by_model.values()):<9} unparsed={unparsed}",
              file=sys.stderr)

    # ---- merge parts ----
    counts = {}
    unparsed_total = 0
    responses_total = 0
    full_path = os.path.join(args.out, "model_summary_full.json")
    with open(full_path, "w", encoding="utf-8") as full:
        full.write("{" + chr(10))
        stem_files = [os.path.join(parts_dir, f"{s}.json") for s in stems]
        for n, pf_path in enumerate(stem_files):
            with open(pf_path, encoding="utf-8") as f:
                part = json.load(f)
            key, models, n_items = part["key"], part["models"], part["n_items"]
            resp = part["responses"]
            unparsed_total += part["unparsed"]
            responses_total += sum(resp.values())
            counts[key] = {
                m: {"n_items": len(v),
                    "coverage": round(100.0 * len(v) / n_items, 1) if n_items else None,
                    "responses": resp.get(m, 0)}
                for m, v in models.items()
            }
            full.write(f"  {json.dumps(key)}: {json.dumps(models)}")
            full.write(("," if n < len(stem_files) - 1 else "") + chr(10))
        full.write("}" + chr(10))

    with open(os.path.join(args.out, "model_summary_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    # Benchmark item totals come from the part files, so the table can show the
    # real denominator alongside the best-covered model's count.
    totals = {}
    for pf_path in [os.path.join(parts_dir, f"{s}.json") for s in stems]:
        with open(pf_path, encoding="utf-8") as f:
            part = json.load(f)
        totals[part["key"]] = part["n_items"]
    with open(os.path.join(args.out, "model_summary_counts.md"), "w", encoding="utf-8") as f:
        f.write(render_table(counts, totals))

    all_models = sorted({m for b in counts.values() for m in b})
    print(f"\ndone: {len(counts)} benchmarks, {len(all_models)} distinct models, "
          f"{responses_total} responses, {unparsed_total} unparsed response_ids",
          file=sys.stderr)
    print(f"  {full_path} ({os.path.getsize(full_path)/1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
