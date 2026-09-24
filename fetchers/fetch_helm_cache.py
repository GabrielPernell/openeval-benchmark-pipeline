#!/usr/bin/env python3
"""
fetch_helm_cache.py
====================
Download HELM run folders into a local cache the HELM loader can read.

Why this exists
---------------
HELM publishes its raw run output in the public `crfm-helm-public` Google Cloud
bucket, so no author contact or credentials are needed. The layout is

    {project}/benchmark_output/runs/{release}/{run_name}/

where one run folder is one model evaluated on one scenario. Everything the
conversion needs is in five of the eight files per folder; `display_*.json` are
pre-rendered for HELM's own web UI and `instances.json` duplicates the
instances already inside `scenario_state.json`.

Run names can't be directory names
----------------------------------
A HELM run name looks like

    commonsense:dataset=openbookqa,method=multiple_choice_joint,model=deepseek-ai_deepseek-v3

and the colon makes that an invalid path on Windows (`WinError 267`). Folders
on disk therefore use a sanitized name, and `manifest.json` in the cache root
maps each sanitized folder back to its real run name. The run name is also
still recorded inside each folder's `run_spec.json`, so nothing is lost.

Usage
-----
python fetch_helm_cache.py --scenario commonsense
python fetch_helm_cache.py --scenario commonsense --release v1.13.0 --project lite
python fetch_helm_cache.py --list-scenarios
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net_utils import get_with_retry  # noqa: E402

BUCKET = "crfm-helm-public"
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
DOWNLOAD_BASE = f"https://storage.googleapis.com/{BUCKET}"

# The files the conversion reads, plus the two small ones worth keeping for
# provenance (run_spec records the real run name; stats has the run-level
# aggregates we check our per-item scores against).
WANTED = [
    "scenario.json",             # benchmark name, tags, definition path
    "scenario_state.json",       # adapter spec, prompts, completions
    "per_instance_stats.json",   # per-item metric values
    "run_spec.json",             # provenance: the run name and its spec
    "stats.json",                # run-level aggregates, for cross-checking
]

# GCS is not HuggingFace: HF_TOKEN must not be attached to these requests.
NO_AUTH: dict = {}


def sanitize(run_name: str) -> str:
    """Make a HELM run name usable as a directory name on any platform."""
    out = run_name
    for bad, good in ((":", "__"), ("=", "-"), (",", "_"), ("/", "-")):
        out = out.replace(bad, good)
    return out


def list_runs(project: str, release: str, scenario: str | None) -> list[str]:
    """Run folder names under a release, optionally filtered to one scenario."""
    prefix = f"{project}/benchmark_output/runs/{release}/"
    if scenario:
        prefix += scenario
    runs, token = [], None
    while True:
        params = {"prefix": prefix, "delimiter": "/", "fields": "prefixes,nextPageToken"}
        if token:
            params["pageToken"] = token
        data = get_with_retry(LIST_API, params, headers=NO_AUTH).json()
        runs += [p.split("/")[-2] for p in data.get("prefixes", [])]
        token = data.get("nextPageToken")
        if not token:
            return sorted(runs)


def scenario_names(project: str, release: str) -> list[str]:
    """The distinct scenario prefixes in a release (the part before the ':')."""
    return sorted({r.split(":")[0] for r in list_runs(project, release, None)})


def fetch_run(project: str, release: str, run: str, out_root: str, force: bool) -> dict:
    dest = os.path.join(out_root, sanitize(run))
    os.makedirs(dest, exist_ok=True)
    sizes = {}
    for name in WANTED:
        path = os.path.join(dest, name)
        if os.path.exists(path) and not force:
            sizes[name] = os.path.getsize(path)
            continue
        url = f"{DOWNLOAD_BASE}/{project}/benchmark_output/runs/{release}/{run}/{name}"
        resp = get_with_retry(url, timeout=120, headers=NO_AUTH)
        with open(path, "wb") as f:
            f.write(resp.content)
        sizes[name] = len(resp.content)
    total = sum(sizes.values())
    print(f"      {sanitize(run):<70} {total/1048576:5.1f} MB", file=sys.stderr)
    return {"run_name": run, "folder": sanitize(run), "bytes": total, "files": sizes}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="lite", help="HELM leaderboard project (default: lite)")
    ap.add_argument("--release", default="v1.13.0", help="release tag (default: v1.13.0)")
    ap.add_argument("--scenario", help="only runs whose name starts with this, e.g. commonsense")
    ap.add_argument("--out", default=None,
                    help="cache root (default: data_cache/helm_<project>_<release>)")
    ap.add_argument("--limit", type=int, help="stop after this many runs")
    ap.add_argument("--force", action="store_true", help="re-download files already cached")
    ap.add_argument("--list-scenarios", action="store_true",
                    help="print the release's scenarios and exit")
    args = ap.parse_args()

    if args.list_scenarios:
        for name in scenario_names(args.project, args.release):
            print(name)
        return

    out_root = args.out or os.path.join("data_cache", f"helm_{args.project}_{args.release}")
    print(f"[1/2] Listing runs in {args.project} {args.release}"
          f"{' for ' + args.scenario if args.scenario else ''}...", file=sys.stderr)
    runs = list_runs(args.project, args.release, args.scenario)
    if not runs:
        sys.exit(f"no runs found for prefix {args.scenario!r} in {args.project} {args.release}")
    if args.limit:
        runs = runs[:args.limit]
    print(f"      -> {len(runs)} run(s)", file=sys.stderr)

    print(f"[2/2] Downloading into {out_root} ...", file=sys.stderr)
    os.makedirs(out_root, exist_ok=True)
    entries = [fetch_run(args.project, args.release, r, out_root, args.force) for r in runs]

    # The manifest is how the loader recovers the real run name from a folder,
    # and how a re-run knows what is already here.
    manifest_path = os.path.join(out_root, "manifest.json")
    manifest = {"project": args.project, "release": args.release,
                "scenario": args.scenario, "runs": entries}
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)

    total = sum(e["bytes"] for e in entries)
    print(f"\ndone: {len(entries)} run(s), {total/1048576:.1f} MB", file=sys.stderr)
    print(f"      manifest: {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
