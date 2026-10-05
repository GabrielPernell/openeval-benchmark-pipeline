#!/usr/bin/env python3
"""
add_visfactor_responses.py
==========================
Attach the VisFactor lab's model responses to the items built by
convert_to_openeval.py, and write a submission folder for the OpenEval HF
dataset (gzipped item-schema JSONL + README):

    submissions/visfactor/visfactor.jsonl.gz
    submissions/visfactor/README.md

Input is the lab's VisFactor.zip (fetchers/fetch_visfactor_cache.py
--responses): one VLMEvalKit result file per model in Vanilla/, standard
prompt only, 3,046 rows each. Older runs are <model>.csv with <model>_acc.csv;
newer ones are <model>.tsv with <model>_score.json and add `raw_prediction`
and `thinking` columns. Rows join to items on `index`, and every row's
question, answer and `additional` must match the item's.

Prompts
-------
`request_input` is the authors' build_prompt (visfactor_scoring.py): the
question with `<br>` as newlines and <ADDITIONAL_n> filled in, split around
the images, which are referenced by viewer-row URL as in the items. The one
exception is Qwen-2-VL-72B: its config entry leaves VLMEvalKit's Qwen prompt
on, which for a VQA-type dataset sends all images first, then the raw
question, plus a fixed suffix (vlmeval/vlm/qwen2_vl/prompt.py,
_build_vqa_prompt). The Qwen2.5-VL entries turn it off.

Generation settings come from the lab's vlmeval/config.py entry for each
model, with the wrapper class's defaults where the entry sets nothing. Models
with no entry there (newer than the config), or with two candidate entries
that disagree (Qwen-2.5-VL-32B), are left null.

Scores
------
One score per response, `correct` (1/0), recomputed with the authors' own
answer extraction (visfactor_scoring.py); the extracted answer is attached.
The benchmark's headline number groups rows: a test question counts only if
every row sharing its (category_id, eval_index) is correct, and the overall
score is the mean over the 20 subtests. The run fails unless the recomputed
per-row marks equal the stored `correct` column and every model's overall
score equals its published one.

Reasoning traces
----------------
Three models (Seed-2.0 Pro/Lite/Mini) come with full reasoning traces, a
median of 22,000-56,000 characters per response depending on the model. The
response data is not public yet, so at the lab's request the traces are left
out for now: wherever a trace exists, the response holds the final answer
followed by an empty string in its place, to be filled in later.

Usage
-----
python -X utf8 add_visfactor_responses.py --items output/visfactor_items.json
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetchers"))

import fetch_visfactor_cache
from loaders.visfactor import VIEWER_ROW_URL, VisFactorAdapter
from response_format import build_response_json
from submission_utils import check_items, counts, write_jsonl_gz
from visfactor_scoring import build_prompt, overall, score_row

csv.field_size_limit(2**31 - 1)

METRIC = "correct"
OUT_DIR = "submissions/visfactor"
OUT_NAME = "visfactor.jsonl.gz"

QWEN_VQA_SUFFIX = "\nPlease try to answer the question with short words or phrases if possible."
QWEN_CUSTOM_PROMPT = {"Qwen-2-VL-72B"}

# From the model's name; closed models have no published size. Kimi-K2.5 and
# GLM-5V-Turbo are left empty because their sizes could not be confirmed.
SIZES = {
    "LLaMA-3.2-11B-Vision": "11B",
    "LLaMA-3.2-90B-Vision": "90B",
    "Qwen-2-VL-72B": "72B",
    "Qwen-2.5-VL-32B": "32B",
    "Qwen-2.5-VL-72B": "72B",
    "Qwen-3-VL-235B-A22B": "235B",
}


def _gen(temperature=None, max_tokens=None, top_p=None, top_k=None, do_sample=None):
    return {"temperature": temperature, "do_sample": do_sample, "top_k": top_k,
            "top_p": top_p, "max_tokens": max_tokens}


# vlmeval/config.py entry -> settings. Defaults of the wrapper classes:
# GPT4V/Claude3V/QwenVLAPI max_tokens 2048; Qwen2VLChat max_new_tokens 2048,
# top_p 0.001, top_k 1, temperature 0.01; llama_vision on an Instruct
# checkpoint do_sample=True, temperature 0.6, top_p 0.9, and max_new_tokens
# 512 for a VQA-type dataset.
GENERATION = {
    "o1-2024-12-17": _gen(0.0, 16384),
    "o3-2025-04-16": _gen(0.0, 16384),
    "o4-Mini-2025-04-16": _gen(0.0, 16384),
    "GPT-4.1-2025-04-14": _gen(0.0, 2048),
    "GPT-4o-2024-11-20": _gen(0.0, 2048),
    "GPT-4o-Mini-2024-07-18": _gen(0.5, 2048),
    "GPT-5-Mini-2025-08-07": _gen(0.0, 2048),
    "GPT-5.1-2025-11-13": _gen(0.0, 2048),
    "Gemini-2.5-Flash": _gen(0.0, 2048),
    "Gemini-2.5-Pro": _gen(0.0, 2048),
    # Two entries (claude-3-5-sonnet-20240620 / -20241022), same settings.
    "Claude-Sonnet-3.5": _gen(0.0, 2048),
    "Claude-Sonnet-3.7": _gen(0.0, 2048),
    "Claude-Sonnet-4": _gen(0.0, 2048),
    # QwenVLMax and QwenVLMax-250408 share these settings.
    "Qwen-VL-Plus": _gen(0.0, 2048),
    "Qwen-VL-Max": _gen(0.0, 2048),
    "Seed-1.5-VL": _gen(0.0, 16384),
    "Seed-1.6-Thinking": _gen(0.0, 16384),
    "Moonshot-V1-128K-Vision": _gen(0.0, 2048),
    "Qwen-2-VL-72B": _gen(0.01, 2048, top_p=0.001, top_k=1),
    "Qwen-2.5-VL-72B": _gen(0.01, 2048, top_p=0.001, top_k=1),
    "LLaMA-3.2-11B-Vision": _gen(0.6, 512, top_p=0.9, do_sample=True),
    "LLaMA-3.2-90B-Vision": _gen(0.6, 512, top_p=0.9, do_sample=True),
}


def read_model(results_dir, model):
    """-> (rows, published overall score)"""
    tsv = os.path.join(results_dir, f"{model}.tsv")
    path = tsv if os.path.exists(tsv) else os.path.join(results_dir, f"{model}.csv")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t" if path == tsv else ","))
    score_json = os.path.join(results_dir, f"{model}_score.json")
    if os.path.exists(score_json):
        with open(score_json, encoding="utf-8") as f:
            published = json.load(f)["Overall"]
    else:
        with open(os.path.join(results_dir, f"{model}_acc.csv"), encoding="utf-8") as f:
            published = float(dict(csv.reader(f))["ALL"])
    return path, rows, published


def list_models(results_dir):
    return sorted({f.rsplit(".", 1)[0] for f in os.listdir(results_dir)
                   if f.endswith((".csv", ".tsv")) and not f.endswith("_acc.csv")})


def image_part(src, k):
    return {"type": "image", "url": VIEWER_ROW_URL.format(src["index"]), "image_index": k}


def request_input(model, src):
    if model in QWEN_CUSTOM_PROMPT:
        return ([image_part(src, k) for k in range(len(src["image"]))]
                + [src["question"] + QWEN_VQA_SUFFIX])
    return [text if kind == "text" else image_part(src, text)
            for kind, text in build_prompt(src["question"], src["additional"], len(src["image"]))]


# Stands in for a reasoning trace until the lab's response data is public.
TRACE_PLACEHOLDER = ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="output/visfactor_items.json",
                    help="items JSON from convert_to_openeval.py --dataset visfactor")
    ap.add_argument("--results", default="data_cache/visfactor/Vanilla",
                    help="the Vanilla/ folder from the lab's VisFactor.zip")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    print("[1/4] Loading items...")
    with open(args.items, encoding="utf-8") as f:
        items = json.load(f)
    by_index = {it["item_content"]["input"][0]["index"]: it for it in items}
    for it in items:
        it["responses"] = []

    print("[2/4] Building responses...")
    problems = []
    report = []
    n_traces = 0
    models = list_models(args.results)
    for model in models:
        path, rows, published = read_model(args.results, model)
        if len(rows) != len(items):
            raise ValueError(f"{model}: {len(rows)} rows, expected {len(items)}")
        marks = []
        for row in rows:
            item = by_index[int(row["index"])]
            src = item["item_content"]["input"][0]
            answer = item["item_content"]["references"][0]["answer"]
            if (row["question"], row["answer"], row["additional"] or "") != \
                    (src["question"], answer, src["additional"] or ""):
                raise ValueError(f"{model} row {row['index']} does not match its item")

            pred, ok = score_row(row["category_id"], row["prediction"], answer, row["additional"])
            if str(ok) != row["correct"]:
                raise AssertionError(f"{model} row {row['index']}: recomputed {ok}, "
                                     f"stored {row['correct']}")
            marks.append(ok)

            content = [row["prediction"]]
            if row.get("thinking") not in (None, "", "[]"):
                content.append(TRACE_PLACEHOLDER)
                n_traces += 1
            item["responses"].append(build_response_json(
                item_id=item["item_id"],
                model_name=model,
                request_input=request_input(model, src),
                response_text=row["prediction"],
                scores=[{"name": METRIC, "models": [],
                         "extra_artifacts": [{"type": "extracted_answer", "content": pred}],
                         "value": int(ok)}],
                generation_parameters=GENERATION.get(model),
                size=SIZES.get(model, ""),
            ))
            item["responses"][-1]["response_content"] = content
        _, ours = overall(rows, marks)
        report.append((model, published, ours))
        if abs(ours - published) > 1e-9:
            problems.append(f"{model}: overall {ours:.6f}, published {published:.6f}")
    print(f"      -> {len(models)} models, {n_traces} responses with a trace placeholder")

    print("[3/4] Checking...")
    print(f"      {'model':<26} {'published':>9} {'ours':>9}")
    for model, published, ours in report:
        print(f"      {model:<26} {published:>9.4f} {ours:>9.4f}"
              + ("" if abs(ours - published) <= 1e-9 else "   <-- differs"))
    problems += check_items(items)

    print("[4/4] Writing the submission folder...")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)
    write_jsonl_gz(items, out_path)
    n_items, n_resp, n_models = counts(items)
    write_readme(args.out, n_items, n_resp, n_models, report, n_traces)
    print(f"      -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB), "
          f"{n_items} items, {n_resp} responses, {n_models} models")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("all checks passed")


def write_readme(out_dir, n_items, n_resp, n_models, report, n_traces):
    a = VisFactorAdapter()
    rows = "\n".join(f"| {m} | {100 * p:.1f} |" for m, p, _ in sorted(report, key=lambda r: -r[1]))
    with_settings = len([m for m, _, _ in report if m in GENERATION])
    text = f"""# VisFactor

Items and model responses from [VisFactor]({a.paper_url}) ("Human Cognitive
Benchmarks Reveal Foundational Visual Gaps in MLLMs"): 20 vision subtests from
the Factor-Referenced Cognitive Tests, digitised as image questions. Items from
[{a.dataset_url.split('datasets/')[1]}]({a.dataset_url}), pinned to revision
`{fetch_visfactor_cache.REVISION}`; responses are the lab's own VLMEvalKit runs
(standard prompt), shared with OpenEval. Evaluation code:
[CUHK-ARISE/VisFactor](https://github.com/CUHK-ARISE/VisFactor).

| items | responses | models |
|---|---|---|
| {n_items:,} | {n_resp:,} | {n_models} |

## Items

`item_content.input` holds every source column under its original name except
`answer`, which is the reference. Images are not embedded: `image` is a list
of `{{"url", "image_index"}}`, where `url` is the item's row in the HF dataset
viewer and `image_index` is the N of the `<IMAGE_N>` tag in `question`.

`question` is a template: `<br>` is a line break and `<ADDITIONAL_n>` slots
are filled from the `;`-separated `additional` column. `eval_index` groups
rows into test questions (see Scores).

## Responses

- **Prompt.** `request_input` is the prompt the authors' code builds
  (`build_prompt` in `vlmeval/dataset/visfactor.py`): the filled-in question,
  split around the images, which are referenced as in the items. Qwen-2-VL-72B
  is the exception: its configuration leaves VLMEvalKit's Qwen prompt on,
  which sends the images first, then the unfilled question with
  "Please try to answer the question with short words or phrases if
  possible." appended.
- **Generation parameters** are taken from the authors' VLMEvalKit
  configuration for the {with_settings} models listed in it, and are null for
  the others.
- **Model size** is filled in where it is known and left empty otherwise.
- **Reasoning traces.** Seed-2.0 Pro, Lite and Mini were run with reasoning
  traces (a median of 22,000-56,000 characters per response, depending on
  the model). They are left out for now: in those {n_traces:,} responses,
  `response_content` is the final answer followed by an empty string where
  the trace will go. No other model's traces were recorded.

## Scores

Each response has one score, `{METRIC}` (1/0), from the authors' answer
extraction; the extracted answer is in `extra_artifacts`. The benchmark's
headline score groups rows: a test question counts only if every row with the
same `category_id` and `eval_index` is correct, a subtest's score is the share
of its test questions that count, and the overall score is the mean over the
20 subtests. Computed that way, every model's overall score equals the one the
lab published:

| model | overall (%) |
|---|---|
{rows}

## Citation

```bibtex
@article{{huang2025human,
  title={{Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs}},
  author={{Huang, Jen-Tse and Dai, Dasen and Huang, Jen-Yuan and Yuan, Youliang and Liu, Xiaoyuan and Wang, Wenxuan and Jiao, Wenxiang and He, Pinjia and Tu, Zhaopeng and Duan, Haodong}},
  journal={{arXiv preprint arXiv:2502.16435}},
  year={{2025}}
}}
```
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
