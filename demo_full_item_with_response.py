#!/usr/bin/env python3
"""
demo_full_item_with_response.py
=================================
NOT part of the automated pipeline -- this is a one-off, hand-built
demonstration answering the question "what does a fully validator-passing
item look like, once responses[] is filled in?"

convert_to_openeval.py deliberately leaves `responses: []` on every item it
produces, because DSR-Bench (like DocHop and VisionWebDev) only released the
benchmark items themselves -- no pre-computed model outputs. validator.py
requires `responses` (and each response's `scores`) to be non-empty, so a
freshly-converted item will always fail validation until someone actually
runs a model on it and records the result.

To show what that looks like, this script takes item #0 from the converted
DSR-Bench output (a short hashmap question) and adds ONE genuine response:
Claude (this session, running as claude-sonnet-5) solved the hashmap
question by hand below, and its answer is scored against DSR-Bench's
ground_truth with an exact-match comparison computed in code (not asserted).

Honesty note on model_adaptation.generation_parameters: this Cowork session
doesn't expose literal sampling settings (temperature/top_k/top_p) to me, so
rather than invent plausible-looking numbers, those are left as `null`
(the schema's "required" presence level explicitly allows null -- see
validator.py's handling of presence == 'required'). `max_tokens` is set to
1500 because the DSR-Bench prompt itself instructs "Answer the question in
1500 tokens."
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validator

ITEMS_PATH = "output/dsr_bench_main_items.json"
OUT_PATH = "output/dsr_bench_demo_item_with_response.json"

with open(ITEMS_PATH) as f:
    items = json.load(f)

item = items[0]
question = item["item_content"]["input"][0]
ground_truth = item["item_content"]["references"][0]
assert "hashmap" in question.lower(), "expected item #0 to be the short hashmap question"

# --- My (Claude's) actual worked solution to this specific question ---
# 10 buckets, bucket = key % 10, starting empty:
#   (add, (31,37))  -> bucket1: [[31,37]]
#   (remove, 31)     -> bucket1: []
#   (add, (53,64))  -> bucket3: [[53,64]]
#   (add, (83,51))  -> bucket3: [[53,64],[83,51]]
#   (add, (91,1))    -> bucket1: [[91,1]]
#   (add, (45,88))  -> bucket5: [[45,88]]
#   (add, (49,45))  -> bucket9: [[49,45]]
# Final buckets 0-9:
my_answer_text = (
    "Bucket 0: []\n"
    "Bucket 1: [[91, 1]]\n"
    "Bucket 2: []\n"
    "Bucket 3: [[53, 64], [83, 51]]\n"
    "Bucket 4: []\n"
    "Bucket 5: [[45, 88]]\n"
    "Bucket 6: []\n"
    "Bucket 7: []\n"
    "Bucket 8: []\n"
    "Bucket 9: [[49, 45]]\n"
    "Final answer: [], [[91, 1]], [], [[53, 64], [83, 51]], [], [[45, 88]], [], [], [], [[49, 45]]"
)
my_final_line = my_answer_text.split("Final answer: ")[-1].strip()

# Score: exact match against ground_truth, computed for real (not asserted).
is_correct = my_final_line == ground_truth.strip()
print(f"My answer's final line : {my_final_line}")
print(f"DSR-Bench ground_truth : {ground_truth}")
print(f"Exact match?            {is_correct}")

response = {
    "response_id": f"{item['item_id']}_claude-sonnet-5_0",
    "model": {
        "name": "claude-sonnet-5",
        "size": None,
        "model_adaptation": {
            "system_instruction": "",
            "generation_parameters": {
                "temperature": None,   # not exposed by this harness -- see module docstring
                "do_sample": None,
                "top_k": None,
                "top_p": None,
                "max_tokens": 1500,     # per the prompt's own "Answer in 1500 tokens" instruction
            },
            "tools": [],
        },
    },
    "item_adaptation": {
        "request_input": item["item_content"]["input"],   # presented to the model unmodified
        "demonstrations": [],
        "external_resources": [],
    },
    "response_content": [{"text": my_answer_text}],
    "scores": [
        {
            "metric": {"name": "exact_match", "models": [], "extra_artifacts": []},
            "value": bool(is_correct),
        }
    ],
}

item_with_response = dict(item)
item_with_response["responses"] = [response]

os.makedirs("output", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(item_with_response, f, indent=2)

ok, violations = validator.validate_entry(item_with_response)
print(f"\nWrote {OUT_PATH}")
print(f"Validator result: {'PASS' if ok else 'FAIL'}")
if not ok:
    for v in violations:
        print(f"  * {v['field']} [{v['violation_type'].__name__}]")
