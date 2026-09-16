#!/usr/bin/env python3
"""
Review checker for Open-Eval-Commons/OpenEval PR #3.

Streams one gzipped JSONL file from the PR branch, decompresses it on the fly
(nothing large is written to disk) and reports:
  1. validator.validate_entry result for EVERY entry (the contributor's claim)
  2. item / response / distinct-model counts, to compare against their table
  3. things the validator cannot see -- empty references, non-str tags,
     score value types, duplicate item_ids
  4. provenance spot-checks -- generation parameters, judges, source ids

Deliberately does NOT print item or response text: these are safety benchmarks
containing adversarial prompts and complying completions. Only structure,
counts, and field names are reported.
"""
import argparse
import collections
import gzip
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MISSING_VALIDATOR = """validator.py is missing. It belongs to open-eval/OpenEval rather than this
repo, so it is not checked in here. Copy validator.py and item_schema.json
from https://github.com/open-eval/OpenEval into the pipeline root, then run
this again."""

try:
    import validator  # noqa: E402
except ModuleNotFoundError:
    raise SystemExit(_MISSING_VALIDATOR)

BASE = ("https://huggingface.co/datasets/Open-Eval-Commons/OpenEval/resolve/"
        "refs%2Fpr%2F3/submissions/metasafetybench/")


def stream_lines(name):
    import requests
    r = requests.get(BASE + name, stream=True, timeout=600,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    seen = {"n": 0, "pct": -1}

    class Counting(io.RawIOBase):
        def __init__(self, it):
            self.it = it
            self.buf = b""

        def readable(self):
            return True

        def readinto(self, b):
            while len(self.buf) < len(b):
                try:
                    c = next(self.it)
                except StopIteration:
                    break
                self.buf += c
                seen["n"] += len(c)
                if total:
                    pct = int(seen["n"] * 100 / total)
                    if pct % 20 == 0 and pct != seen["pct"]:
                        print(f"    ... {pct}% ({seen['n']/1e6:.0f}/{total/1e6:.0f} MB)",
                              file=sys.stderr)
                        seen["pct"] = pct
            if not self.buf:
                return 0
            n = min(len(b), len(self.buf))
            b[:n] = self.buf[:n]
            self.buf = self.buf[n:]
            return n

    raw = io.BufferedReader(Counting(r.iter_content(1024 * 256)))
    with gzip.GzipFile(fileobj=raw) as gz:
        for line in io.TextIOWrapper(gz, encoding="utf-8"):
            line = line.strip()
            if line:
                yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="e.g. harmbench.jsonl.gz")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--dump-ids", default=None, help="write item ids + source ids to this JSON")
    args = ap.parse_args()

    n_items = n_resp = 0
    violations = collections.Counter()
    bad_items = []
    models = collections.Counter()
    judges = collections.Counter()
    score_types = collections.Counter()
    gen_params = collections.Counter()
    tag_types = collections.Counter()
    empty_refs = empty_inputs = 0
    ids = collections.Counter()
    source_ids = []
    src_meta_keys = collections.Counter()
    versions = collections.Counter()
    bench_names = collections.Counter()
    resp_per_item = []
    no_score = 0
    parse_errors = 0

    for i, line in enumerate(stream_lines(args.file)):
        if args.max_items and n_items >= args.max_items:
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        n_items += 1

        ok, v = validator.validate_entry(item)
        if not ok:
            for x in v:
                violations[f"{x['field']}:{x['violation_type'].__name__}"] += 1
            if len(bad_items) < 5:
                bad_items.append((item.get("item_id"), [x["field"] for x in v]))

        ids[item.get("item_id")] += 1
        src = item.get("item_metadata", {}).get("source", {})
        bench_names[src.get("benchmark_name")] += 1
        versions[src.get("benchmark_version")] += 1
        if src.get("source_item_id") is not None:
            source_ids.append(src["source_item_id"])
        sm = src.get("source_item_metadata")
        if isinstance(sm, dict):
            for k in sm:
                src_meta_keys[k] += 1
        for t in src.get("benchmark_tags") or []:
            tag_types[type(t).__name__] += 1

        content = item.get("item_content", {})
        if not content.get("references") or not str(content["references"][0] or "").strip():
            empty_refs += 1
        if not content.get("input"):
            empty_inputs += 1

        rs = item.get("responses") or []
        n_resp += len(rs)
        resp_per_item.append(len(rs))
        for r in rs:
            models[r.get("model", {}).get("name")] += 1
            gp = r.get("model", {}).get("model_adaptation", {}).get("generation_parameters", {})
            for k, val in gp.items():
                gen_params[f"{k}={val!r}"] += 1
            sc = r.get("scores") or []
            if not sc:
                no_score += 1
            for s in sc:
                judges[s.get("metric", {}).get("name")] += 1
                score_types[type(s.get("value")).__name__] += 1

    print(f"\n{'='*66}\nFILE: {args.file}\n{'='*66}")
    print(f"items                 {n_items}")
    print(f"responses             {n_resp}")
    print(f"distinct models       {len(models)}")
    print(f"responses/item        min={min(resp_per_item)} max={max(resp_per_item)} "
          f"mean={sum(resp_per_item)/len(resp_per_item):.1f}")
    print(f"json parse errors     {parse_errors}")
    print(f"\n-- 1. validator.validate_entry --")
    print(f"   items failing      {sum(1 for _ in bad_items) if bad_items else 0}"
          f"{' (showing first 5)' if bad_items else '  <- ALL PASS'}")
    for b in bad_items:
        print(f"      {b}")
    if violations:
        print(f"   violations         {dict(violations)}")
    print(f"\n-- 2. things the validator cannot see --")
    print(f"   duplicate item_ids {sum(c-1 for c in ids.values() if c > 1)}")
    print(f"   empty references   {empty_refs}")
    print(f"   empty inputs       {empty_inputs}")
    print(f"   benchmark_tag types {dict(tag_types)}")
    print(f"   score value types  {dict(score_types)}")
    print(f"   responses w/o score {no_score}")
    print(f"\n-- 3. provenance --")
    print(f"   benchmark_name     {dict(bench_names)}")
    print(f"   benchmark_version  {dict(versions)}")
    print(f"   judges             {dict(judges)}")
    print(f"   generation params  {dict(gen_params)}")
    print(f"   source_item_id     {len(source_ids)}/{n_items} present")
    print(f"   source_item_metadata keys {dict(src_meta_keys)}")
    print(f"   top models         {[m for m, _ in models.most_common(3)]}")

    if args.dump_ids:
        json.dump({"item_ids": list(ids), "source_item_ids": source_ids},
                  open(args.dump_ids, "w", encoding="utf-8"))
        print(f"\n   wrote ids -> {args.dump_ids}")


if __name__ == "__main__":
    main()
