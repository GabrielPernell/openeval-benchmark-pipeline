#!/usr/bin/env python3
"""
convert_to_openeval.py
========================
Reusable CLI pipeline: raw benchmark rows -> OpenEval item schema JSON.

Usage examples
--------------
# Convert a cached sample (works offline):
python3 convert_to_openeval.py --dataset dsr_bench --subset main \
    --mode cache --source data_cache/dsr_bench_sample_raw.json \
    --out output/dsr_bench_main_items.json \
    --contributor-name "Your Name" --contributor-email "you@example.edu"

# Convert straight from HuggingFace (needs network access):
python3 convert_to_openeval.py --dataset dsr_bench --subset main \
    --mode hf --limit 500 --out output/dsr_bench_main_items.json \
    --contributor-name "Your Name" --contributor-email "you@example.edu"

# Validate an already-converted file without re-converting:
python3 convert_to_openeval.py --validate-only output/dsr_bench_main_items.json

Adding a new dataset
---------------------
See adapter_base.py's docstring. In short: write loaders/<name>.py with a
DatasetAdapter subclass, then register it in get_adapter() below. All four
loaders are implemented and verified against real data. helm is the only one
whose source also carries model responses; see add_helm_responses.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_utils import build_item, generate_time, save_json, load_json
from loaders.dsr_bench import DSRBenchAdapter
from loaders.dochop import DocHopAdapter
from loaders.visionwebdev import VisionWebDevAdapter
from loaders.helm import HELMAdapter
_MISSING_VALIDATOR = """validator.py is missing. It belongs to open-eval/OpenEval rather than this
repo, so it is not checked in here. Copy validator.py and item_schema.json
from https://github.com/open-eval/OpenEval into the pipeline root, then run
this again."""

try:
    import validator
except ModuleNotFoundError:
    raise SystemExit(_MISSING_VALIDATOR)


def get_adapter(dataset: str, subset: str):
    if dataset == "dsr_bench":
        return DSRBenchAdapter(subset=subset)
    if dataset == "dochop":
        return DocHopAdapter()
    if dataset == "helm":
        # HELM's subsets are its scenarios; the run folders are named after
        # the scenario group, so "commonsense" selects the OpenBookQA runs.
        if subset == "main":
            raise ValueError(
                "--subset is required for helm: a scenario name such as commonsense. "
                "List what a release has with "
                "fetchers/fetch_helm_cache.py --list-scenarios."
            )
        return HELMAdapter(subset=subset)
    if dataset == "visionwebdev":
        # VisionWebDev's subsets are its three task types (webpage/frontend/
        # website); "main" is the DSR-Bench default and means nothing here.
        if subset == "main":
            raise ValueError(
                "--subset is required for visionwebdev: one of webpage (Level 1), "
                "frontend (Level 2), website (Level 3)."
            )
        return VisionWebDevAdapter(subset=subset)
    raise ValueError(f"Unknown --dataset '{dataset}'. Expected one of: dsr_bench, dochop, visionwebdev, helm")


def run_validation(items: list) -> int:
    """Run validator.validate_entry on every item, print a report, return violation count."""
    n_bad = 0
    for i, item in enumerate(items):
        ok, violations = validator.validate_entry(item)
        if not ok:
            n_bad += 1
            print(f"\n--- Item #{i} ({item.get('item_id', '?')}) has {len(violations)} violation(s) ---")
            for v in violations:
                print(f"  * {v['field']}  [{v['violation_type'].__name__}]  -- {v['field_desc']}")
    print(f"\nValidation summary: {len(items) - n_bad}/{len(items)} items passed, {n_bad} item(s) had violations.")
    return n_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["dsr_bench", "dochop", "visionwebdev", "helm"],
                    help="Which dataset adapter to use")
    ap.add_argument("--subset", default="main",
                    help="Dataset-specific subset/config. DSR-Bench: main|challenge|spatial|"
                         "natural|code (default: main). VisionWebDev: webpage|frontend|website. "
                         "HELM: a scenario name such as commonsense.")
    ap.add_argument("--mode", choices=["cache", "hf"], default="cache", help="'cache' = local JSON file, 'hf' = live network fetch")
    ap.add_argument("--source", default=None, help="Path to local cache file (required if --mode cache)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of rows to convert")
    ap.add_argument("--out", default=None, help="Output JSON path for converted items")
    ap.add_argument("--contributor-name", default="Anonymous")
    ap.add_argument("--contributor-email", default="")
    ap.add_argument("--contributor-affiliation", default="")
    ap.add_argument("--validate-only", default=None, metavar="PATH", help="Skip conversion; just validate an existing items JSON file")
    args = ap.parse_args()

    if args.validate_only:
        items = load_json(args.validate_only)
        n_bad = run_validation(items)
        sys.exit(1 if n_bad else 0)

    if not args.dataset:
        ap.error("--dataset is required unless --validate-only is used")
    if args.mode == "cache" and not args.source:
        ap.error("--source is required when --mode cache")
    if not args.out:
        ap.error("--out is required")

    adapter = get_adapter(args.dataset, args.subset)
    contributor = {
        "name": args.contributor_name,
        "email": args.contributor_email,
        "affiliation": args.contributor_affiliation,
    }

    print(f"[1/4] Loading rows for {adapter.benchmark_name} ({args.mode} mode)...")
    rows = list(adapter.load_rows(args.source, limit=args.limit, mode=args.mode))
    print(f"      -> loaded {len(rows)} raw rows")

    print("[2/4] Mapping rows into OpenEval items...")
    ingestion_time = generate_time()
    items = [
        build_item(row, adapter, idx, ingestion_time, contributor, responses=[])
        for idx, row in enumerate(rows)
    ]
    print(f"      -> built {len(items)} items (responses=[] -- see README.md)")

    print(f"[3/4] Writing output to {args.out}...")
    save_json(items, args.out)

    print("[4/4] Validating output against item_schema.json...")
    run_validation(items)


if __name__ == "__main__":
    main()
