#!/usr/bin/env python3
"""
add_ltb_responses.py
====================
Attach the Last Translation Benchmark's model translations to the items built
by convert_to_openeval.py, and write a submission folder for the OpenEval HF
dataset (gzipped item-schema JSONL + README):

    submissions/last_translation_benchmark/last_translation_benchmark.jsonl.gz
    submissions/last_translation_benchmark/README.md

Every source example carries its own `translations` list: one entry per model
plus a "human" reference. The human entry is already in the item's
references (loaders/ltb.py); every other entry becomes a response here.

Prompts
-------
The release does not include prompts. The authors' code
(github.com/zouharvi/last-translation-benchmark, server/services.py
`translate_openrouter` and scripts/20b-translate_by_extra_models.py
`get_prompt`) builds the one prompt every LLM received, and `prompt_for`
below reproduces it character for character -- including the quoted
instructions and the "Translate the provide ..." wording for media-only
examples, which differ slightly from Prompt 1 as printed in the paper.
Two groups of systems did not get that prompt:
  * TRANSLATION_APIS -- Google Translate and Lara are called with the text,
    the language pair and (Lara only) the instructions; there is no prompt.
  * UNPUBLISHED -- translation models submitted to the leaderboard by other
    groups, and two legacy systems; how they were prompted is not in the
    authors' code, so request_input holds only what the model translated.

Generation settings: the code passes only `seed=0`, so temperature, top_p,
top_k, max_tokens and reasoning effort are the providers' defaults. They are
recorded as null (not set), not guessed.

Scores
------
One score per response, `passes_all_rules` (1/0): a translation passes only
if it meets every one of the example's rules, which is the paper's headline
measure. The per-rule verdicts are kept as an extra artifact. The released
verdicts are Gemini 3.1 Pro's (03a-prepare_release.py exports
verified_extra["Gemini 3.1 Pro"]; the server's verifier is
google/gemini-3.1-pro-preview).

Checks
------
Response counts against the source, unique response ids, validator on every
item, and each model's pass rate on LTBv1-eval against the Gemini 3.1 Pro
verifier column of the paper's Main Table 2.

Usage
-----
python -X utf8 add_ltb_responses.py --items output/ltb_items.json
"""
import argparse
import collections
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetchers"))

import fetch_ltb_cache

_MISSING_VALIDATOR = """validator.py is missing. It belongs to open-eval/OpenEval rather than this
repo, so it is not checked in here. Copy validator.py and item_schema.json
from https://github.com/open-eval/OpenEval into the pipeline root, then run
this again."""

try:
    import validator
except ModuleNotFoundError:
    raise SystemExit(_MISSING_VALIDATOR)
from loaders.ltb import LTBAdapter, media_mime
from response_format import build_response_json

VERIFIER = "google/gemini-3.1-pro-preview"
METRIC = "passes_all_rules"

TRANSLATION_APIS = {"Google Translate", "Lara"}

# Submitted to the leaderboard from outside (scripts/40-mt_to_leaderboard.py
# lists them, nothing in the repo runs them), plus NLLB 54B and QwenMT, which
# appear in the data but nowhere in the code.
UNPUBLISHED = {"TranslateGemma", "Tower+", "GemmaX2-28-9B", "HY-MT2", "Seed-X-PPO-7B",
               "NLLB 3.3B", "Command A Translate", "NLLB 54B", "QwenMT"}

# Every other model is an LLM that went through the prompt below. Most are
# listed in the authors' code; Gemini 2.5 Pro, Gemini 3.5 Flash, Cohere
# Command A, Llama 4 Scout and GPT-4.1 Nano are earlier platform models no
# longer listed, but the platform's only LLM path is the same function.

# Sizes from the model's name or the id the authors call it by
# (e.g. google/gemma-4-31b-it, nvidia/nemotron-3-ultra-550b-a55b), and for
# Llama 4 and Command A from their public model cards. Unknown sizes stay "".
SIZES = {
    "Gemma 4": "31B",
    "Nemotron 3 Ultra": "550B",
    "gpt-oss-20b": "20B",
    "Voxtral Small": "24B",
    "Seed-X-PPO-7B": "7B",
    "GemmaX2-28-9B": "9B",
    "NLLB 3.3B": "3.3B",
    "NLLB 54B": "54B",
    "Llama 4 Maverick": "400B",
    "Llama 4 Scout": "109B",
    "Command A": "111B",
    "Cohere Command A": "111B",
}

# Main Table 2, Gemini 3.1 Pro verifier column: % of LTBv1-eval examples
# passing all rules, excluding examples the model did not translate.
PAPER_PASS_RATE = {
    "human": 99.9, "Gemini 3.1 Pro": 39.3, "GPT-5.6 Sol": 28.1, "GPT-5.6 Luna": 15.4,
    "Gemini 3.5 Flash Lite": 11.9, "GPT-5.6 Terra": 12.2, "Kimi K3": 11.7,
    "Deepseek V4 Pro": 10.8, "Qwen 3.7 Plus": 10.2, "Gemini 2.5 Flash": 6.9, "Gemma 4": 7.1,
    "Nemotron 3 Ultra": 4.8, "Qwen 3.7 Flash": 5.8, "TranslateGemma": 5.0,
    "Claude Sonnet 4.5": 4.5, "GPT-5.4 Mini": 2.7, "HY-MT2": 2.2, "Llama 4 Maverick": 2.0,
    "GemmaX2-28-9B": 2.1, "Seed-X-PPO-7B": 1.8, "Command A+": 1.4, "Tower+": 2.1,
    "Google Translate": 1.1, "Claude Haiku 4.5": 0.7, "Lara": 0.8,
    "Command A Translate": 0.7, "gpt-oss-20b": 1.3, "NLLB 3.3B": 1.3, "Command A": 0.7,
    "TinyAya Global": 0.7,
}

OUT_DIR = "submissions/last_translation_benchmark"
OUT_NAME = "last_translation_benchmark.jsonl.gz"


def context_type(mime):
    return "audio" if "audio" in mime else ("video" if "video" in mime else "image")


def prompt_for(row):
    """The authors' translate_openrouter / get_prompt, verbatim."""
    text = row["source_text"]
    src, tgt = row["source_lang"], row["target_lang"]
    if not row["source_media"]:
        prompt = (f"Translate the following text from {src} to {tgt}. "
                  f"Output only the translation and nothing else:\n{text}")
    else:
        ctx = context_type(row["source_media"].split(",")[0])
        if text:
            prompt = (f"Translate the following text from {src} to {tgt}. "
                      f"Use the provided {ctx} as additional context. "
                      f"Output only the translation and nothing else:\n{text}")
        else:
            prompt = (f"Translate the provide {ctx} from {src} to {tgt}. "
                      f"Output only the textual translation and nothing else.")
    if row["source_instructions"]:
        prompt += f'\nAdditional instructions for this translation are: "{row["source_instructions"]}"'
    return prompt


def media_part(row, item):
    """The media as the item references it (a viewer URL, not the base64)."""
    media = item["item_content"]["input"][0]["source_media"]
    return {"type": context_type(media["mime_type"]), **media}


def request_input(model, row, item):
    if model in TRANSLATION_APIS:
        call = {"text": row["source_text"], "source_lang": row["source_lang"],
                "target_lang": row["target_lang"]}
        if model == "Lara" and row["source_instructions"]:
            call["instructions"] = row["source_instructions"]
        parts = [call]
    elif model in UNPUBLISHED:
        parts = [row["source_text"]] if row["source_text"] else []
    else:
        parts = [prompt_for(row)]
    if row["source_media"]:
        parts.append(media_part(row, item))
    return parts


def build(items, rows):
    by_id = {r["id"]: r for r in rows}
    skipped = []
    n_resp = 0
    for item in items:
        row = by_id[item["item_content"]["input"][0]["id"]]
        trials = collections.Counter()
        responses = []
        for t in row["translations"]:
            model = t["model"]
            if model == "human":
                continue
            trial = trials[model]
            trials[model] += 1
            verdicts = t["verified"]
            if not verdicts or any(v is None for v in verdicts):
                skipped.append((row["id"], model, verdicts))
                continue
            if len(verdicts) != len(row["verification_rules"]):
                raise ValueError(f"example {row['id']} / {model}: "
                                 f"{len(verdicts)} verdicts for {len(row['verification_rules'])} rules")
            responses.append(build_response_json(
                item_id=item["item_id"],
                model_name=model,
                request_input=request_input(model, row, item),
                response_text=t["translation"] if t["translation"] is not None else "",
                scores=[{"name": METRIC, "models": [VERIFIER],
                         "extra_artifacts": [{"type": "rule_verdicts", "content": list(verdicts)}],
                         "value": int(all(verdicts))}],
                trial=trial,
                size=SIZES.get(model, ""),
            ))
        item["responses"] = responses
        n_resp += len(responses)
    return n_resp, skipped


def pass_rates(rows):
    """Per model, % of LTBv1-eval examples passing all rules (first entry per
    model; examples a model did not translate are excluded, as in the paper)."""
    out = collections.defaultdict(list)
    for row in rows:
        if "LTBv1-eval" not in row["tags"]:
            continue
        seen = set()
        for t in row["translations"]:
            if t["model"] in seen or not t["verified"] or None in t["verified"]:
                continue
            seen.add(t["model"])
            out[t["model"]].append(all(t["verified"]))
    return {m: 100 * sum(v) / len(v) for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="output/ltb_items.json",
                    help="items JSON from convert_to_openeval.py --dataset ltb")
    ap.add_argument("--cache", default="data_cache/ltb/v1.json")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    print("[1/4] Loading items and the LTB release...")
    with open(args.items, encoding="utf-8") as f:
        items = json.load(f)
    rows = fetch_ltb_cache.load(args.cache)
    if len(items) != len(rows):
        raise ValueError(f"{len(items)} items but {len(rows)} source examples")
    expected = sum(t["model"] != "human" for r in rows for t in r["translations"])

    print("[2/4] Building responses...")
    n_resp, skipped = build(items, rows)
    print(f"      -> {n_resp} responses ({expected} model translations in the source, "
          f"{len(skipped)} left out for a missing verdict: {skipped})")
    if n_resp + len(skipped) != expected:
        raise AssertionError("response count does not add up")

    print("[3/4] Checking...")
    problems = []
    ids = [r["response_id"] for it in items for r in it["responses"]]
    if len(ids) != len(set(ids)):
        problems.append(f"{len(ids) - len(set(ids))} duplicate response ids")
    vio = collections.Counter()
    for it in items:
        ok, vios = validator.validate_entry(it)
        for v in vios:
            vio[f"{v['field']} ({v['violation_type'].__name__})"] += 1
    problems += [f"validator: {k} x{n}" for k, n in sorted(vio.items())]

    ours = pass_rates(rows)
    print(f"      {'model':<24} {'paper':>6} {'ours':>6}")
    worst = 0.0
    for model, paper in PAPER_PASS_RATE.items():
        diff = ours[model] - paper
        worst = max(worst, abs(diff))
        print(f"      {model:<24} {paper:>6.1f} {ours[model]:>6.1f}"
              + ("   <-- differs" if abs(diff) >= 0.15 else ""))
    print(f"      largest difference: {worst:.2f} points")

    print("[4/4] Writing the submission folder...")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    models = {r["model"]["name"] for it in items for r in it["responses"]}
    texts = [r["response_content"][0] for it in items for r in it["responses"]]
    write_readme(args.out, len(items), n_resp, len(models), skipped,
                 n_empty=sum(t == "" for t in texts),
                 n_teapot=sum(t == '"teapot"' for t in texts))
    print(f"      -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB), "
          f"{len(items)} items, {n_resp} responses, {len(models)} models")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("all checks passed")


def write_readme(out_dir, n_items, n_resp, n_models, skipped, n_empty, n_teapot):
    a = LTBAdapter()
    skipped_txt = "; ".join(f"example {i} / {m}" for i, m, _ in skipped) or "none"
    text = f"""# Last Translation Benchmark (LTBv1)

Items and model responses from the
[Last Translation Benchmark]({a.paper_url}): human-authored, peer-reviewed
examples (text, images, audio, video) that break state-of-the-art translation
models across many language pairs, each with handcrafted verification rules
that a correct translation must meet. Source data:
[{a.dataset_url.split('datasets/')[1]}]({a.dataset_url}) (CC BY 4.0), pinned to
revision `{fetch_ltb_cache.REVISION}`.

| items | responses | models |
|---|---|---|
| {n_items:,} | {n_resp:,} | {n_models} |

## Items

`item_content.input` holds every source column under its original name except
`translations` and `verification_rules`. `item_content.references` holds
`verification_rules` and the human reference translation (the `translations`
entry with `model: "human"`, including its verdicts).

Media (images, audio, video) is not embedded: `source_media` is a URL to the
example's row in the HF dataset viewer, with its `mime_type`.

`tags` includes `LTBv1-eval` for the 911 examples the paper's leaderboard is
computed on.

## Responses

One response per model translation. Where a model has two translations of the
same example, they are trials 0 and 1.

- **Prompt.** `request_input` is the prompt the authors' code sent
  (`translate_openrouter` in the LTB repository), followed by the media when
  there is any. Google Translate and Lara are translation APIs with no prompt;
  their `request_input` is the call's parameters. For TranslateGemma, Tower+,
  GemmaX2-28-9B, HY-MT2, Seed-X-PPO-7B, NLLB 3.3B, NLLB 54B, Command A Translate
  and QwenMT the prompt is not published, and `request_input` is the source
  text (and media) only.
- **Generation parameters.** The authors' code sets only `seed=0`; everything
  else is the provider's default, recorded as null.
- **Model size** is filled in where it is known and left empty otherwise.

## Scores

Each response has one score, `{METRIC}`: 1 if the translation meets every
verification rule, else 0 -- the paper's headline measure. The per-rule
verdicts are in the score's `extra_artifacts` (`rule_verdicts`, one boolean
per rule, in the order of `verification_rules`). Verdicts are by
`{VERIFIER}`, as released.

Averaging `{METRIC}` over the `LTBv1-eval` examples reproduces the paper's
leaderboard (Main Table 2, Gemini 3.1 Pro verifier).

## Notes

- Left out for a missing verdict in the source: {skipped_txt}.
- {n_empty} translations are empty strings and {n_teapot} are the literal `"teapot"`, the
  authors' placeholder for output that degenerated into repetition. Both are
  kept as released, with their verdicts.

## Citation

```bibtex
@misc{{zouhar2026translationbenchmark,
  title={{Last Translation Benchmark}},
  author={{LTB}},
  year={{2026}},
  eprint={{2609.04173}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CL}},
  url={{https://arxiv.org/abs/2609.04173}},
}}
```
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
