#!/usr/bin/env python3
"""
add_helm_responses.py
======================
Attach HELM's model responses to items already converted by
`convert_to_openeval.py --dataset helm`.

HELM is the first benchmark in this pipeline whose source ships the responses
alongside the items, so this is the step that makes a converted benchmark
validate cleanly instead of failing on the schema's non-empty `responses`.

One run folder is one model on one scenario, so N runs of the same scenario
produce N responses per item.

What this fixes in the upstream converter
-----------------------------------------
`helm_converter.py` carries eight "TODO: manual check before every run"
markers. Three of them are wrong for anything but a summarization scenario and
were silently producing bad output:

  * the metric list is hardcoded to rouge/summac/BERTScore, none of which
    OpenBookQA emits -- every response would have got an empty `scores`;
  * `metric.models` is hardcoded to the summac and DeBERTa scorer models,
    which have nothing to do with exact match;
  * demonstrations are recovered by splitting the prompt and removing
    `adapter_spec.instructions`, but the prompt's preamble is
    `global_prefix` + `instructions`. Stripping only the second leaves the
    first behind as a bogus sixth "demonstration". Here a demonstration is
    instead identified by carrying the adapter's `output_prefix`, and the
    count is asserted against `max_train_instances`, so the check the TODO
    asks a human to do every run happens automatically.

`system_instruction` gets the whole preamble for the same reason.

Metric selection
----------------
HELM records 27 stats per instance, most of them instrumentation rather than
measurements of the answer (`num_prompt_tokens`, `inference_runtime`,
`batch_size`, `training_co2_cost`, `finish_reason_*`, `prompt_truncated`).
Only the scoring metrics below go into `scores`; a schema field described as an
evaluation metric is the wrong home for a CO2 estimate.

Each stat is a distribution (count/sum/min/max/mean/variance/stddev) while the
schema's `value` is a single number, so `mean` is used. Every selected metric
was verified to have `count == 1`, where mean, min and max coincide.

Usage
-----
python add_helm_responses.py --items output/helm_openbookqa_items.json \\
    --cache data_cache/helm_lite_v1.13.0 --scenario commonsense \\
    --out output_with_responses
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loaders.helm import HELMAdapter, read_manifest  # noqa: E402
from response_format import (build_response_json, json_to_parquet_row,  # noqa: E402
                             write_responses)

# The scoring metrics, chosen over HELM's 27 per-instance stats.
SCORING_METRICS = [
    "exact_match", "exact_match@5",
    "quasi_exact_match", "quasi_exact_match@5",
    "prefix_exact_match", "prefix_exact_match@5",
    "quasi_prefix_exact_match", "quasi_prefix_exact_match@5",
    "max_prob", "logprob",
]

# HELM's own headline metric, used for the cross-check against stats.json.
PRIMARY_METRIC = "exact_match"

_SIZE_RE = re.compile(r"[0-9.]+b")


def model_size(name: str) -> str:
    """HELM records no size field, so it is read off the name when present."""
    match = _SIZE_RE.search(name.lower())
    return match.group(0) if match else ""


def preamble(adapter_spec: dict) -> str:
    """Everything the model saw before the first demonstration."""
    return (adapter_spec.get("global_prefix", "") or "") + \
           (adapter_spec.get("instructions", "") or "")


def demonstrations_for(req: dict, adapter_spec: dict) -> list:
    """The few-shot examples in this request's prompt.

    A demonstration is a question chunk that carries its own answer, so the
    instruction preamble drops out without having to be matched and stripped.
    """
    prompt = req["request"]["prompt"]
    text = req["instance"]["input"]["text"]
    head = prompt.split(adapter_spec["input_prefix"] + text)[0]
    out_prefix = adapter_spec.get("output_prefix", "") or "\n"
    return [c.strip() for c in head.split(adapter_spec["input_prefix"]) if out_prefix in c]


def scores_for(stats_row: dict) -> list:
    """The selected metrics for one instance, as schema score dicts."""
    by_name = {s["name"]["name"]: s for s in stats_row["stats"]}
    out = []
    for name in SCORING_METRICS:
        stat = by_name.get(name)
        if stat is None:
            continue
        out.append({
            "name": name,
            # Exact match needs no scoring model. Upstream hardcodes the
            # summac/DeBERTa models here, which belong to other metrics.
            "models": [],
            "extra_artifacts": [],
            "value": stat["mean"],
        })
    return out


def load_run(cache_root: str, folder: str) -> tuple:
    run_dir = os.path.join(cache_root, folder)
    with open(os.path.join(run_dir, "scenario_state.json"), encoding="utf-8") as f:
        state = json.load(f)
    with open(os.path.join(run_dir, "per_instance_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(run_dir, "stats.json"), encoding="utf-8") as f:
        run_stats = json.load(f)
    return state, stats, run_stats


def build_id_map(items: list, cache_root: str, scenario: str) -> dict:
    """{HELM instance id: our item_id}, checked against the items' own text.

    The adapter yields rows in sorted instance-id order and build_item numbers
    items in that same order, so position joins them. That is an assumption
    about two separate code paths, so it is verified rather than trusted.
    """
    adapter = HELMAdapter(subset=scenario)
    rows = adapter.load_rows(cache_root, mode="cache")
    if len(rows) != len(items):
        sys.exit(f"{len(rows)} rows from the cache but {len(items)} items in the JSON. "
                 "Re-run convert_to_openeval.py against this same cache.")
    id_map = {}
    for row, item in zip(rows, items):
        if item["item_content"]["input"][0] != row["question"]:
            sys.exit(f"item {item['item_id']} does not line up with HELM instance "
                     f"{row['helm_instance_id']}; the orders have diverged.")
        id_map[row["helm_instance_id"]] = item["item_id"]
    return id_map


def responses_for_run(state: dict, stats: list, id_map: dict) -> list:
    adapter_spec = state["adapter_spec"]
    model_name = adapter_spec["model"].split("/")[-1]
    system_instruction = preamble(adapter_spec)
    expect_demos = adapter_spec.get("max_train_instances")

    stats_by_key = {(s["instance_id"], s["train_trial_index"]): s for s in stats}
    trials = collections.Counter()
    out = []

    for req in state["request_states"]:
        inst = req["instance"]
        helm_id = inst["id"]
        if helm_id not in id_map:
            sys.exit(f"HELM instance {helm_id} has no converted item; "
                     "the items and the cache do not match.")

        demos = demonstrations_for(req, adapter_spec)
        if expect_demos is not None and len(demos) != expect_demos:
            sys.exit(f"extracted {len(demos)} demonstrations for {helm_id} but the adapter "
                     f"says max_train_instances={expect_demos}. The prompt layout changed; "
                     "check demonstrations_for() before trusting this run.")

        stats_row = stats_by_key.get((helm_id, req["train_trial_index"]))
        if stats_row is None:
            sys.exit(f"no per-instance stats for {helm_id} "
                     f"(train_trial_index={req['train_trial_index']})")

        completions = req["result"]["completions"]
        text = completions[0]["text"] if completions else ""

        item_id = id_map[helm_id]
        trial = trials[(item_id, model_name)]
        trials[(item_id, model_name)] += 1

        out.append(build_response_json(
            item_id=item_id,
            model_name=model_name,
            request_input=[req["request"]["prompt"]],
            response_text=text,
            scores=scores_for(stats_row),
            generation_parameters={
                "temperature": adapter_spec.get("temperature"),
                "do_sample": bool(adapter_spec.get("temperature")),
                "top_k": req["request"].get("top_k_per_token"),
                "top_p": req["request"].get("top_p"),
                "max_tokens": adapter_spec.get("max_tokens"),
            },
            system_instruction=system_instruction,
            trial=trial,
            size=model_size(model_name),
            demonstrations=demos,
        ))
    return model_name, out


def published_primary(run_stats: list) -> float | None:
    """The run-level mean of the primary metric, as HELM publishes it."""
    for s in run_stats:
        name = s.get("name", {})
        if name.get("name") == PRIMARY_METRIC and name.get("split") == "test" \
                and "perturbation" not in name:
            return s.get("mean")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="items JSON from convert_to_openeval.py")
    ap.add_argument("--cache", required=True, help="HELM cache root from fetch_helm_cache.py")
    ap.add_argument("--scenario", default="commonsense", help="scenario prefix to attach")
    ap.add_argument("--out", default="output_with_responses", help="output directory")
    ap.add_argument("--slug", default=None,
                    help="parquet filename stem (default: the items' benchmark name)")
    args = ap.parse_args()

    with open(args.items, encoding="utf-8") as f:
        items = json.load(f)
    print(f"[1/5] {len(items)} items from {args.items}")

    manifest = read_manifest(args.cache)
    runs = [e["folder"] for e in manifest["runs"] if e["folder"].startswith(args.scenario)]
    if not runs:
        sys.exit(f"no runs for scenario {args.scenario!r} in {args.cache}")
    print(f"[2/5] {len(runs)} run(s) for {args.scenario} "
          f"({manifest.get('project')} {manifest.get('release')})")

    id_map = build_id_map(items, args.cache, args.scenario)

    print("[3/5] Building responses...")
    by_item = collections.defaultdict(list)
    checks = []
    for folder in runs:
        state, stats, run_stats = load_run(args.cache, folder)
        model_name, responses = responses_for_run(state, stats, id_map)
        for r in responses:
            by_item[r["response_id"].rsplit("_", 2)[0]].append(r)
        ours = [s["value"] for r in responses for s in
                [next((x for x in r["scores"] if x["metric"]["name"] == PRIMARY_METRIC), None)]
                if s]
        mine = sum(ours) / len(ours) if ours else None
        checks.append((model_name, mine, published_primary(run_stats), len(responses)))
        print(f"      {model_name:<18} {len(responses):>5} responses")

    # Cross-check against HELM's own aggregates, the way DocHop's conversion was
    # checked against the authors' published accuracies.
    print(f"[4/5] Cross-checking mean {PRIMARY_METRIC} against each run's stats.json...")
    bad = []
    for model_name, mine, published, n in checks:
        if published is None:
            print(f"      {model_name:<18} ours={mine!r}  published=(absent) -- skipped")
            continue
        ok = mine is not None and abs(mine - published) < 1e-9
        print(f"      {model_name:<18} ours={mine:.6f}  published={published:.6f}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(model_name)
    if bad:
        sys.exit(f"mean {PRIMARY_METRIC} does not reproduce HELM's published value for: "
                 f"{', '.join(bad)}")

    slug = args.slug or items[0]["item_metadata"]["source"]["benchmark_name"].lower()
    total = 0
    for item in items:
        item["responses"] = by_item.get(item["item_id"], [])
        total += len(item["responses"])
    missing = [i["item_id"] for i in items if not i["responses"]]
    if missing:
        sys.exit(f"{len(missing)} item(s) ended up with no responses, e.g. {missing[:3]}")

    print(f"[5/5] Writing {total} responses...")
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f"{slug}_items_with_responses.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(items, f, indent=2)

    parquet_rows = [json_to_parquet_row(r) for item in items for r in item["responses"]]
    paths = write_responses(parquet_rows, os.path.join(args.out, "response"), slug)

    print(f"\ndone: {total} responses over {len(items)} items, "
          f"{len({r['model']['name'] for i in items for r in i['responses']})} models")
    print(f"      {json_path} ({os.path.getsize(json_path)/1048576:.1f} MB)")
    for p in paths:
        print(f"      {p} ({os.path.getsize(p)/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
