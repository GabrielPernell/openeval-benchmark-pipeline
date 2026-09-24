#!/usr/bin/env python3
"""
check_repo_join.py
===================
Verify that the exported tables actually join before anything is submitted.

Why this exists
---------------
Item ids embed the conversion's timestamp, and response ids embed the item id.
So re-running a conversion without re-running the export silently leaves two
tables that look fine on their own and join to nothing. That has now happened
twice: once on DSR-Bench (0 of 51,850 responses joined) and once on OpenBookQA,
after a re-run for an unrelated regression check.

Nothing about a broken join is visible in either file, so it needs its own
check. Run this before opening a PR.

Checks, per benchmark:
  * every response's embedded item id exists in the item table
  * every item's `item_metadata.source` resolves to a bench/ row
  * ids are unique within each table

Usage
-----
python check_repo_join.py
python check_repo_join.py --export repo_export --responses output_with_responses
"""
import argparse
import glob
import os
import re
import sys

import pyarrow.parquet as pq

SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.parquet$")


def slug_of(path: str) -> str:
    return SHARD_RE.sub("", os.path.basename(path))


def read_all(paths: list, columns=None) -> list:
    rows = []
    for p in sorted(paths):
        rows += pq.read_table(p, columns=columns).to_pylist()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="repo_export", help="dir holding item/ and bench/")
    ap.add_argument("--responses", default="output_with_responses",
                    help="dir holding response/")
    args = ap.parse_args()

    bench_paths = glob.glob(os.path.join(args.export, "bench", "*.parquet"))
    if not bench_paths:
        sys.exit(f"no bench/ table under {args.export}")
    bench_names = {r["benchmark_name"] for r in read_all(bench_paths)}
    print(f"bench/: {len(bench_names)} benchmark(s) -> {', '.join(sorted(bench_names))}\n")

    item_files = {}
    for p in glob.glob(os.path.join(args.export, "item", "*.parquet")):
        item_files.setdefault(slug_of(p), []).append(p)
    resp_files = {}
    for p in glob.glob(os.path.join(args.responses, "response", "*.parquet")):
        resp_files.setdefault(slug_of(p), []).append(p)

    if not item_files:
        sys.exit(f"no item/ tables under {args.export}")

    problems = []
    print(f"{'benchmark':<16}{'items':>8}{'responses':>11}{'joined':>9}  source -> bench/")
    for slug in sorted(item_files):
        items = read_all(item_files[slug], columns=["item_id", "item_metadata"])
        item_ids = {r["item_id"] for r in items}
        if len(item_ids) != len(items):
            problems.append(f"{slug}: duplicate item_ids "
                            f"({len(items)} rows, {len(item_ids)} distinct)")

        sources = {r["item_metadata"]["source"] for r in items}
        dangling = sorted(s for s in sources if s not in bench_names)
        src_note = "ok" if not dangling else f"DANGLING {dangling}"
        if dangling:
            problems.append(f"{slug}: item source {dangling} matches no bench/ row")

        n_resp = joined = 0
        if slug in resp_files:
            resps = read_all(resp_files[slug], columns=["response_id"])
            rids = {r["response_id"] for r in resps}
            if len(rids) != len(resps):
                problems.append(f"{slug}: duplicate response_ids "
                                f"({len(resps)} rows, {len(rids)} distinct)")
            n_resp = len(resps)
            joined = sum(1 for r in resps
                         if r["response_id"].rsplit("_", 2)[0] in item_ids)
            if joined != n_resp:
                problems.append(f"{slug}: only {joined}/{n_resp} responses join to an item "
                                "-- re-export with --preserve-ids-from")
        print(f"{slug:<16}{len(items):>8}{n_resp:>11}"
              f"{(str(joined) if n_resp else '-'):>9}  {src_note}")

    orphan = sorted(set(resp_files) - set(item_files))
    for slug in orphan:
        problems.append(f"{slug}: response table with no item table")

    print()
    if problems:
        print(f"FAILED with {len(problems)} problem(s):")
        for p in problems:
            print("  *", p)
        sys.exit(1)
    print("all tables join, all sources resolve, no duplicate ids")


if __name__ == "__main__":
    main()
