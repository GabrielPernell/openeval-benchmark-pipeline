#!/usr/bin/env python3
"""
add_dochop_responses.py
========================
Attaches the DocHop authors' released model results to our converted items.

Input is the `dochop-release-results/` directory shared by the authors: one
folder per model, each containing

    <Model>_DocHop.xlsx       2,074 rows: index, original_id, question, answer,
                              task, chart_num, depth, image_path, prediction
    <Model>_DocHop_acc.csv    the aggregate accuracy table

The xlsx has no `correct` column -- only the raw prediction -- so scores are
recomputed with the authors' own heuristic (ported verbatim in
dochop_scoring.py). This is checked, not assumed: the per-model accuracy we
compute must reproduce their published `_acc.csv` figure exactly, and the run
fails if any model disagrees.

Items are joined on `original_id`, which our converted items already carry as
`source_item_id`, so no text matching is needed.

Outputs
-------
  <out>/dochop_items_with_responses.json   our nested JSON format
  <out>/response/dochop-*.parquet          the HF dataset's response tables

Usage
-----
python add_dochop_responses.py --results <dir> --items output/dochop_items.json \
    --out output_with_responses
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from response_format import build_response_json, json_to_parquet_row, write_responses

METRIC_NAME = "heuristic_accuracy"

# The repo already lists this model in lowercase. Matching the existing
# spelling avoids adding another case-variant duplicate of a model that is
# already there -- the problem the model summary surfaced. Every other model
# keeps the name the authors published.
NAME_ALIASES = {"Gemini-2.5-Flash": "gemini-2.5-flash"}


def load_scoring():
    """dochop_scoring.py is a verbatim port of the authors' heuristic."""
    from dochop_scoring import is_correct, extract_answer
    return is_correct, extract_answer


def read_model_dir(path):
    from openpyxl import load_workbook
    xlsx = glob.glob(os.path.join(path, "*.xlsx"))
    if not xlsx:
        raise FileNotFoundError(f"no .xlsx in {path}")
    wb = load_workbook(xlsx[0], read_only=True, data_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    idx = {c: header.index(c) for c in
           ("original_id", "question", "answer", "prediction", "image_path")}
    rows = []
    for row in it:
        if all(v is None for v in row):
            continue
        rows.append({k: row[i] for k, i in idx.items()})
    wb.close()

    published = None
    for c in glob.glob(os.path.join(path, "*_acc.csv")):
        with open(c, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["Dimension"] == "Overall":
                    published = float(r["Score"])
                    break
    return rows, published


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dochop-release-results directory")
    ap.add_argument("--items", default="output/dochop_items.json")
    ap.add_argument("--cache", default="data_cache/dochop_raw.json")
    ap.add_argument("--out", default="output_with_responses")
    ap.add_argument("--limit-models", type=int, default=None)
    args = ap.parse_args()

    is_correct, extract_answer = load_scoring()
    os.makedirs(args.out, exist_ok=True)

    items = json.load(open(args.items, encoding="utf-8"))

    # Our JSON items don't carry source_item_id (only the parquet export does),
    # but each one records the source row it came from in its image pointer.
    # The cache maps that row index back to the original_id the results use.
    cache = json.load(open(args.cache, encoding="utf-8"))
    index_to_source = {r["index"]: str(r["original_id"]) for r in cache}

    by_source_id = {}
    for it in items:
        it.setdefault("responses", [])
        row_index = None
        for entry in it["item_content"]["input"]:
            if isinstance(entry, dict) and isinstance(entry.get("content"), dict):
                row_index = entry["content"].get("row_index")
                break
        src = index_to_source.get(row_index)
        if src is None:
            raise ValueError(f"cannot resolve source id for item {it['item_id']}")
        by_source_id[src] = it
    if len(by_source_id) != len(items):
        raise ValueError(f"{len(items)} items collapsed to {len(by_source_id)} source ids")
    print(f"loaded {len(items)} items, all joinable by original_id", file=sys.stderr)

    model_dirs = sorted(d for d in os.listdir(args.results)
                        if os.path.isdir(os.path.join(args.results, d)))
    if args.limit_models:
        model_dirs = model_dirs[:args.limit_models]

    parquet_rows = []
    mismatches, unmatched_total, ref_mismatch = [], 0, 0
    for d in model_dirs:
        rows, published = read_model_dir(os.path.join(args.results, d))
        model_name = NAME_ALIASES.get(d, d)
        n = correct = unmatched = 0
        for r in rows:
            item = by_source_id.get(str(r["original_id"]))
            if item is None:
                unmatched += 1
                continue
            extracted = extract_answer(r["prediction"])
            ok = is_correct(r["answer"], extracted)
            n += 1
            correct += int(ok)

            # The answer in the released results should equal the reference we
            # converted from the dataset; a mismatch means the two disagree.
            if str(r["answer"]).strip() != str(item["item_content"]["references"][0]).strip():
                ref_mismatch += 1
            resp = build_response_json(
                item_id=item["item_id"],
                model_name=model_name,
                request_input=[json.dumps({"role": "user", "content": r["question"]},
                                          ensure_ascii=False)],
                response_text=str(r["prediction"]) if r["prediction"] is not None else "",
                scores=[{"name": METRIC_NAME, "models": [], "value": int(ok),
                         "extra_artifacts": [{"type": "extracted_answer",
                                              "content": extracted}]}],
                external_resources=[{"type": "document_image",
                                     "content": r["image_path"]}],
            )
            item["responses"].append(resp)
            parquet_rows.append(json_to_parquet_row(resp))

        unmatched_total += unmatched
        ours = 100.0 * correct / n if n else 0.0
        agree = published is not None and abs(ours - published) < 1e-6
        if not agree:
            mismatches.append((d, ours, published))
        print(f"  {model_name:<28} {n:>5} responses  acc {ours:>6.2f}% "
              f"(published {published:.2f}%) {'OK' if agree else 'MISMATCH'}"
              + (f"  unmatched={unmatched}" if unmatched else ""), file=sys.stderr)

    if mismatches:
        print("\nFAILED: recomputed accuracy does not match the authors' published "
              f"figures for {len(mismatches)} model(s): {mismatches[:3]}", file=sys.stderr)
        sys.exit(1)
    if ref_mismatch:
        print(chr(10) + f"WARNING: {ref_mismatch} response(s) whose released answer "
              "differs from the reference we converted", file=sys.stderr)
    if unmatched_total:
        print(f"\nWARNING: {unmatched_total} prediction(s) had no matching item",
              file=sys.stderr)

    items_out = os.path.join(args.out, "dochop_items_with_responses.json")
    with open(items_out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    paths = write_responses(parquet_rows, os.path.join(args.out, "response"), "dochop")

    print(f"\nwrote {items_out} ({os.path.getsize(items_out)/1e6:.1f} MB)")
    print(f"wrote {len(paths)} response parquet shard(s), {len(parquet_rows)} responses")


if __name__ == "__main__":
    main()
