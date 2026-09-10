#!/usr/bin/env python3
"""
Print one item from either side of PR #3, so the two can be compared directly.

  python pr_review/show_item.py --pr harmbench.jsonl.gz --id harmbench_20260810T175017Z_hb_std_0000
  python pr_review/show_item.py --main harmbench --id harmbench_20260421T041055Z_0

--pr streams the PR's gzipped JSONL (nothing saved to disk unless you pass
--cache); --main reads the parquet already on the dataset's main branch.
Responses are summarised rather than dumped, since a single item can carry
190 of them. Pass --full-responses to see them.
"""
import argparse
import gzip
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_submission import stream_lines  # noqa: E402

MAIN = "https://huggingface.co/datasets/Open-Eval-Commons/OpenEval/resolve/main/item/"


def from_pr(fname, item_id, cache=None):
    if cache and os.path.exists(cache):
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            lines = f
            for line in lines:
                obj = json.loads(line)
                if obj.get("item_id") == item_id:
                    return obj
        return None
    for line in stream_lines(fname):
        obj = json.loads(line)
        if obj.get("item_id") == item_id:
            return obj
    return None


def from_main(stem, item_id):
    import fsspec
    import pyarrow.parquet as pq
    import requests
    tree = requests.get(
        "https://huggingface.co/api/datasets/Open-Eval-Commons/OpenEval/tree/main/item",
        timeout=60, headers={"User-Agent": "Mozilla/5.0"}).json()
    files = [e["path"] for e in tree if e["path"].split("/")[-1].startswith(stem + "-")]
    fs = fsspec.filesystem("http")
    for f in files:
        url = "https://huggingface.co/datasets/Open-Eval-Commons/OpenEval/resolve/main/" + f
        for row in pq.ParquetFile(fs.open(url, "rb")).read().to_pylist():
            if row.get("item_id") == item_id:
                return row
    return None


def summarise(item, full_responses=False):
    out = json.loads(json.dumps(item, default=str))
    rs = out.get("responses")
    if isinstance(rs, list) and rs and not full_responses:
        first = rs[0]
        rc = first.get("response_content")
        if isinstance(rc, list) and rc:
            first["response_content"] = [str(rc[0])[:300] + " ...[truncated]"]
        out["responses"] = [first, f"...and {len(rs) - 1} more responses (use --full-responses)"]
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pr", help="PR filename, e.g. harmbench.jsonl.gz")
    g.add_argument("--main", help="benchmark stem on main, e.g. harmbench")
    ap.add_argument("--id", required=True)
    ap.add_argument("--cache", default=None, help="local .jsonl.gz to read instead of streaming")
    ap.add_argument("--full-responses", action="store_true")
    args = ap.parse_args()

    item = from_pr(args.pr, args.id, args.cache) if args.pr else from_main(args.main, args.id)
    if item is None:
        print(f"item_id {args.id!r} not found", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(summarise(item, args.full_responses), indent=2)[:6000])


if __name__ == "__main__":
    main()
