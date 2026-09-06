#!/usr/bin/env python3
"""
export_repo_parquet.py
=======================
Export our converted benchmarks in the layout the OpenEval HF dataset uses
(item/ parquet + a bench/ registry), rather than the flat JSON this pipeline
produces for local validation.

Runs the adapters over the ORIGINAL sources rather than re-reading our own
output/*.json: our per-item taxonomy is flattened into benchmark_tags there, and
reversing that is lossy. Sources are the local caches where we have them and a
live HF fetch for DSR-Bench.

Usage
-----
python export_repo_parquet.py --out repo_export \
    --contributor-name "..." --contributor-email "..." --contributor-affiliation "..."
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repo_format import build_repo_item, write_items, write_bench
from schema_utils import generate_time
from loaders.dsr_bench import DSRBenchAdapter, SUBSETS as DSR_SUBSETS
from loaders.visionwebdev import VisionWebDevAdapter, SUBSETS as VWD_SUBSETS
from loaders.dochop import DocHopAdapter

# benchmark slug -> (list of (adapter, source, mode), bench/ registry row)
def plan(args):
    return {
        "dsr_bench": (
            [(DSRBenchAdapter(subset=s), None, "hf") for s in DSR_SUBSETS],
            {
                "benchmark_name": "DSR-Bench",
                # Subsets live on the items, so there is no meaningful version
                # here -- the repo leaves this empty for most benchmarks.
                "benchmark_version": "",
                "paper_url": "https://arxiv.org/abs/2505.24069",
                "dataset_url": "https://huggingface.co/collections/vitercik-lab/dsr-bench-6826381f6297ff1499134163",
                "benchmark_tags": ["reasoning", "data-structures", "algorithms", "synthetic"],
            },
        ),
        "visionwebdev": (
            [(VisionWebDevAdapter(subset=s),
              f"data_cache/visionwebdev_{s}_raw.json", "cache") for s in VWD_SUBSETS],
            {
                "benchmark_name": "VisionWebDev",
                "benchmark_version": "",
                "paper_url": "https://arxiv.org/abs/2603.26648",
                "dataset_url": "https://huggingface.co/datasets/zai-org/Vision2Web",
                "benchmark_tags": ["web-development", "code-generation", "vision-language",
                                   "agent-verification", "multimodal"],
            },
        ),
        "dochop": (
            [(DocHopAdapter(), "data_cache/dochop_raw.json", "cache")],
            {
                "benchmark_name": "DocHop",
                "benchmark_version": "",
                "paper_url": "https://openreview.net/forum?id=PQFkScoGqz",
                "dataset_url": "https://huggingface.co/datasets/zhuoranyu336/dochop",
                "benchmark_tags": ["multi-hop-reasoning", "document-understanding",
                                   "chart-reasoning", "vision-language", "multimodal"],
            },
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="repo_export")
    ap.add_argument("--only", default=None, help="convert just one benchmark slug")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--contributor-name", default="Anonymous")
    ap.add_argument("--contributor-email", default="")
    ap.add_argument("--contributor-affiliation", default="")
    args = ap.parse_args()

    contributor = {"name": args.contributor_name, "email": args.contributor_email,
                   "affiliation": args.contributor_affiliation}
    ingestion_time = generate_time()
    bench_rows = []
    todo = plan(args)
    if args.only:
        todo = {args.only: todo[args.only]}

    for slug, (sources, bench_row) in todo.items():
        items = []
        for adapter, source, mode in sources:
            label = getattr(adapter, "subset", "-")
            print(f"[{slug}/{label}] loading ({mode})...", file=sys.stderr)
            rows = list(adapter.load_rows(source, limit=args.limit, mode=mode))
            # idx continues across subsets so item_ids stay unique per benchmark
            for row in rows:
                items.append(build_repo_item(row, adapter, len(items), ingestion_time, contributor))
            print(f"[{slug}/{label}] -> {len(rows)} rows (running total {len(items)})", file=sys.stderr)

        paths = write_items(items, os.path.join(args.out, "item"), slug)
        print(f"[{slug}] wrote {len(items)} items -> {[os.path.basename(p) for p in paths]}")
        bench_rows.append(bench_row)

    bench_path = os.path.join(args.out, "bench", "train-00000-of-00001.parquet")
    write_bench(bench_rows, bench_path)
    print(f"[bench] wrote {len(bench_rows)} registry row(s) -> {bench_path}")


if __name__ == "__main__":
    main()
