#!/usr/bin/env python3
"""
fetch_vision2web_cache.py
==========================
Build a local cache file for loaders/visionwebdev.py.

Why this exists
---------------
Vision2Web's HF repo exposes each task type two ways:
  * `<task_type>/test.parquet` -- a manifest (task_name, level, counts,
    prototype filenames, and a TRUNCATED prompt/prd preview). Cheap, but has
    no workflow.json, i.e. no ground truth.
  * `archives/<task_type>.tar.gz` -- the real task directories, including
    workflow.json and the full prompt.txt/prd.md. 431 MB - 2.6 GB each.

This script takes the manifest (via the pipeline's shared net_utils fetcher)
and merges it with the text files streamed out of the archive, writing a
compact JSON cache -- typically a few MB, because the prototype images and
resources are read off the wire and discarded rather than saved.

The archive still has to be downloaded in full to reach the last task in it
(gzip streams are not seekable), but nothing large is written to disk. Use
--max-tasks to stop early when you only want a sample.

Usage
-----
python fetch_vision2web_cache.py --subset webpage --out data_cache/visionwebdev_webpage_raw.json
python fetch_vision2web_cache.py --subset frontend --max-tasks 5 --out data_cache/sample.json
"""
import argparse
import io
import json
import os
import sys
import tarfile

# The pipeline modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net_utils import fetch_hf_rows, get_with_retry
from loaders.visionwebdev import REPO_ID, SUBSETS

ARCHIVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/archives/{{subset}}.tar.gz"

# Text files worth keeping; everything else in the archive is streamed past.
TEXT_FILES = {"workflow.json", "prompt.txt", "prd.md"}


class _StreamReader(io.RawIOBase):
    """Adapts a requests iter_content generator to a file-like object, so
    tarfile can read the archive without it ever landing on disk."""

    def __init__(self, chunks, on_progress=None):
        self._chunks = chunks
        self._buf = b""
        self.bytes_read = 0
        self._on_progress = on_progress

    def readable(self):
        return True

    def readinto(self, b):
        while len(self._buf) < len(b):
            try:
                chunk = next(self._chunks)
            except StopIteration:
                break
            self._buf += chunk
            self.bytes_read += len(chunk)
            if self._on_progress:
                self._on_progress(self.bytes_read)
        if not self._buf:
            return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def stream_archive(subset: str, max_tasks: int | None = None) -> dict:
    """Stream <subset>.tar.gz, returning {task_name: {prompt/prd/workflow/prototypes}}."""
    url = ARCHIVE_URL.format(subset=subset)
    resp = get_with_retry(url, timeout=300, stream=True)
    total_bytes = int(resp.headers.get("content-length", 0))

    state = {"last_pct": -1}

    def progress(n):
        if not total_bytes:
            return
        pct = int(n * 100 / total_bytes)
        if pct != state["last_pct"] and pct % 5 == 0:
            print(f"      ... {pct}% ({n / 1e6:.0f}/{total_bytes / 1e6:.0f} MB)", file=sys.stderr)
            state["last_pct"] = pct

    reader = _StreamReader(resp.iter_content(chunk_size=1024 * 256), on_progress=progress)
    tasks: dict = {}

    with tarfile.open(fileobj=io.BufferedReader(reader), mode="r|gz") as tf:
        for member in tf:
            parts = member.name.split("/")
            # Expected: <task_type>/<task_name>/<...>
            if len(parts) < 3 or not member.isfile():
                continue
            task_name = parts[1]
            if task_name.startswith("."):
                # Archives can ship VCS/metadata directories alongside the task
                # dirs (website.tar.gz carries a .git/), which are not tasks.
                continue
            rest = parts[2:]
            entry = tasks.setdefault(task_name, {"prototypes": [], "resources_seen": 0})

            if rest[0] == "prototypes" and len(rest) == 2:
                entry["prototypes"].append(rest[1])
            elif rest[0] == "resources":
                entry["resources_seen"] += 1
            elif rest[-1] in TEXT_FILES and len(rest) == 1:
                raw = tf.extractfile(member).read().decode("utf-8", errors="replace")
                if rest[0] == "workflow.json":
                    entry["workflow"] = json.loads(raw)
                elif rest[0] == "prompt.txt":
                    entry["prompt"] = raw
                elif rest[0] == "prd.md":
                    entry["prd"] = raw

            if max_tasks and len(tasks) > max_tasks:
                # One task past the target means the previous ones are complete.
                tasks.pop(task_name, None)
                print(f"      ... stopping early at {len(tasks)} tasks (--max-tasks)", file=sys.stderr)
                break

    print(f"      -> streamed {reader.bytes_read / 1e6:.0f} MB, kept text for {len(tasks)} tasks",
          file=sys.stderr)
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=list(SUBSETS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="Stop after roughly this many tasks (sampling; avoids a full download)")
    args = ap.parse_args()

    print(f"[1/3] Fetching {args.subset} manifest from datasets-server...")
    manifest = {r["task_name"]: r for r in fetch_hf_rows(REPO_ID, config=args.subset, split="test")}
    print(f"      -> {len(manifest)} tasks in manifest")

    print(f"[2/3] Streaming archives/{args.subset}.tar.gz (text only, nothing large saved)...")
    streamed = stream_archive(args.subset, max_tasks=args.max_tasks)

    rows = []
    missing_workflow = []
    for task_name, extracted in streamed.items():
        m = manifest.get(task_name, {})
        if "workflow" not in extracted:
            # No workflow.json means no ground truth, which would produce an
            # item with empty references that still passes validation. Drop it
            # loudly rather than emitting something unscoreable.
            missing_workflow.append(task_name)
            continue
        wf = extracted.get("workflow") or []
        computed_cases = sum(len(w.get("content", [])) for w in wf)
        # The manifest's num_test_cases does NOT always match the number of
        # test cases actually in workflow.json (e.g. frontend/anomali: manifest
        # says 5, the file contains 10). workflow.json is the ground truth the
        # harness executes, so counts are computed from it and the manifest's
        # figure is retained separately rather than silently overriding.
        row = {
            "task_name": task_name,
            "task_type": args.subset,
            "level": m.get("level", SUBSETS[args.subset][0]),
            "workflow_steps": len(wf) if wf else m.get("workflow_steps", 0),
            "num_test_cases": computed_cases if wf else m.get("num_test_cases", 0),
            "manifest_num_test_cases": m.get("num_test_cases"),
            "manifest_workflow_steps": m.get("workflow_steps"),
            "resources_count": m.get("resources_count", extracted["resources_seen"]),
            # Prefer the manifest's prototype list (canonical order); the
            # streamed list is a cross-check.
            "prototypes": m.get("prototypes") or sorted(extracted["prototypes"]),
            "workflow": extracted.get("workflow"),
        }
        if "prompt" in extracted:
            row["prompt"] = extracted["prompt"]
        if "prd" in extracted:
            row["prd"] = extracted["prd"]
        rows.append(row)

    rows.sort(key=lambda r: r["task_name"])

    if missing_workflow:
        print(f"      ! WARNING: dropped {len(missing_workflow)} task(s) with no "
              f"workflow.json: {missing_workflow[:5]}", file=sys.stderr)
    if not args.max_tasks and len(rows) != len(manifest):
        print(f"      ! WARNING: streamed {len(rows)} tasks but the manifest lists "
              f"{len(manifest)} -- cache is incomplete.", file=sys.stderr)

    print(f"[3/3] Writing {len(rows)} rows to {args.out}...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"      -> {size_mb:.1f} MB cache written")


if __name__ == "__main__":
    main()
