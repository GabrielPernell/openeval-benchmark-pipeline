#!/usr/bin/env python3
"""
normalize_model_names.py
=========================
The same model appears in OpenEval under several spellings, because different
contributions recorded `model.name` differently. Left alone they count as
separate models, inflating the model count.

Two kinds of collision exist, and they are NOT equally safe to merge:

  case-only    Phi-4 / phi-4, DeepSeek-R1 / deepseek-r1
               -> purely mechanical, merged by default.

  punctuation  Qwen1.5-0.5B-Chat / qwen-1.5-0.5b-chat
               -> same model in all observed cases, but this is a judgement
                  about model identity rather than a formatting fix, so it is
                  only applied with --aggressive.

Reads the per-benchmark part files written by build_model_summary.py and writes
new outputs alongside them; nothing existing is modified.

Merging unions the item sets of the spellings involved, so a model that was run
on different items under different names ends up with its true coverage. The
audit file records every merge and flags any items covered under more than one
spelling (which would mean genuinely duplicated responses).
"""
import argparse
import collections
import glob
import json
import os
import re

from summary_table import render_table


def canonical(name: str, aggressive: bool) -> str:
    """Lowercase always; also drop separators when --aggressive."""
    return re.sub(r"[^a-z0-9]", "", name.lower()) if aggressive else name.lower()


def preferred(spellings):
    """Pick the display name: the lowercase one if present, else the shortest."""
    lower = [s for s in spellings if s == s.lower()]
    return sorted(lower or spellings, key=len)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="model_summary/parts")
    ap.add_argument("--out", default="model_summary")
    ap.add_argument("--aggressive", action="store_true",
                    help="also merge names differing by punctuation (the Qwen family)")
    ap.add_argument("--tag", default=None,
                    help="filename tag (default: normalized / normalized_aggressive)")
    args = ap.parse_args()

    tag = args.tag or ("normalized_aggressive" if args.aggressive else "normalized")
    part_paths = sorted(glob.glob(os.path.join(args.parts, "*.json")))
    parts = [json.load(open(p, encoding="utf-8")) for p in part_paths]

    # Pass 1: build ONE canonical name map across every benchmark. Most
    # collisions are cross-benchmark (Phi-4 in one, phi-4 in another), so
    # grouping per benchmark would leave the global model count unchanged and
    # could even give the same model different display names in different rows.
    all_names = {n for part in parts for n in part["models"]}
    groups = collections.defaultdict(list)
    for name in all_names:
        groups[canonical(name, args.aggressive)].append(name)
    display_of = {}
    for ckey, spellings in groups.items():
        d = preferred(spellings)
        for s_ in spellings:
            display_of[s_] = d

    merged_all, counts, audit = {}, {}, {}
    n_before = set(all_names)
    n_after = set(display_of.values())

    # Pass 2: apply the global map, unioning item sets where names collapse.
    for part in parts:
        key, models, n_items = part["key"], part["models"], part["n_items"]
        by_display = collections.defaultdict(list)
        for name in models:
            by_display[display_of[name]].append(name)

        out = {}
        for display, spellings in sorted(by_display.items()):
            if len(spellings) == 1:
                out[display] = models[spellings[0]]
                continue
            sets = [set(models[s_]) for s_ in spellings]
            union = set().union(*sets)
            overlap = sum(len(a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
            out[display] = sorted(union)
            audit.setdefault(key, []).append({
                "canonical": display,
                "merged_from": sorted(spellings),
                "items_per_spelling": {s_: len(models[s_]) for s_ in spellings},
                "items_after_merge": len(union),
                "items_covered_by_more_than_one_spelling": overlap,
            })

        merged_all[key] = out
        counts[key] = {m: {"n_items": len(v),
                           "coverage": round(100.0 * len(v) / n_items, 1) if n_items else None}
                       for m, v in out.items()}

    # Record the full name map too, so the merge is auditable and reversible.
    audit["_name_map"] = {k: v for k, v in sorted(display_of.items()) if k != v}

    with open(os.path.join(args.out, f"model_summary_{tag}_full.json"), "w", encoding="utf-8") as f:
        json.dump(merged_all, f)
    with open(os.path.join(args.out, f"model_summary_{tag}_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    with open(os.path.join(args.out, f"model_name_merges_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    totals = {part["key"]: part["n_items"] for part in parts}
    with open(os.path.join(args.out, f"model_summary_{tag}_counts.md"), "w", encoding="utf-8") as f:
        f.write(render_table(counts, totals))

    groups_only = {k: v for k, v in audit.items() if k != "_name_map"}
    dup = sum(g["items_covered_by_more_than_one_spelling"]
              for gs in groups_only.values() for g in gs)
    print(f"[{tag}] distinct model names: {len(n_before)} -> {len(n_after)}")
    print(f"[{tag}] merge groups applied: {sum(len(v) for v in groups_only.values())} "
          f"across {len(groups_only)} benchmarks")
    print(f"[{tag}] renamed spellings: {len(audit['_name_map'])}")
    print(f"[{tag}] items covered under more than one spelling: {dup}")


if __name__ == "__main__":
    main()
