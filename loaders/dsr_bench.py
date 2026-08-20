"""
loaders/dsr_bench.py
=====================
Adapter for DSR-Bench ("Evaluating the Structural Reasoning Abilities of LLMs
via Data Structures", arXiv 2505.24069, github.com/dransyhe/DSR-Bench).

Data availability (checked 2026-08-15): publicly released, no gating, on
HuggingFace under the `vitercik-lab` org as five separate configs/subsets:
    DSR-Bench-main, DSR-Bench-challenge, DSR-Bench-spatial,
    DSR-Bench-natural, DSR-Bench-code
Each row has 10 columns: question_id, category, task, operation, question,
ground_truth, prompt, length, release_date, removal_date.

Two ways to get rows:
  1. `mode="hf"` (default): live fetch via the HuggingFace datasets-server
     REST API (https://datasets-server.huggingface.co/rows?...). This is the
     path a normal, unrestricted environment should use. NOTE: as of this
     writing, this sandbox's outbound network proxy blocks huggingface.co
     (`curl` returns 403), so `mode="hf"` will raise a clear error here --
     it is included so the script is genuinely reusable outside this sandbox.
  2. `mode="cache"`: read rows from a local JSON file (a list of row dicts).
     Used to demo/test the pipeline in this sandbox with a small sample
     fetched manually ahead of time (see data_cache/dsr_bench_sample_raw.json).
"""
import json
from adapter_base import DatasetAdapter
from net_utils import fetch_hf_rows

# The five subsets do NOT share one schema. `main` is the richest; the others
# drop columns and add their own, and `code` even uses a different split name:
#   main       question:str  category task operation length prompt   split=test
#   challenge  question:str  category task operation                 split=test
#   spatial    question:str           task operation length dimension(int)
#                                                              split=test
#   natural    question:str           task               length      split=test
#   code       question:LIST          task               mode        split=code
# row_to_content() therefore builds tags from whichever columns are present
# rather than assuming main's set.
SUBSET_SPLITS = {
    "main": "test",
    "challenge": "test",
    "spatial": "test",
    "natural": "test",
    "code": "code",
}

SUBSETS = {
    "main": "vitercik-lab/DSR-Bench-main",
    "challenge": "vitercik-lab/DSR-Bench-challenge",
    "spatial": "vitercik-lab/DSR-Bench-spatial",
    "natural": "vitercik-lab/DSR-Bench-natural",
    "code": "vitercik-lab/DSR-Bench-code",
}


class DSRBenchAdapter(DatasetAdapter):
    def __init__(self, subset: str = "main"):
        if subset not in SUBSETS:
            raise ValueError(f"Unknown DSR-Bench subset '{subset}', expected one of {list(SUBSETS)}")
        self.subset = subset
        repo_id = SUBSETS[subset]
        super().__init__(
            benchmark_name="DSR-Bench",
            # There's no formal semantic-version number for DSR-Bench; the most
            # meaningful "version" is which of the 5 released subsets/configs
            # this item came from, since each has different item distributions.
            benchmark_version=subset,
            paper_url="https://arxiv.org/abs/2505.24069",
            dataset_url=f"https://huggingface.co/datasets/{repo_id}",
            base_tags=["reasoning", "data-structures", "algorithms", "synthetic"],
        )
        self.repo_id = repo_id

    def load_rows(self, source: str, limit: int | None = None, mode: str = "cache"):
        if mode == "cache":
            with open(source) as f:
                rows = json.load(f)
            return rows[:limit] if limit else rows

        if mode == "hf":
            # Live network path -- paginated + rate-limit-aware (see net_utils).
            return fetch_hf_rows(self.repo_id, config="default",
                                 split=SUBSET_SPLITS[self.subset], limit=limit)

        raise ValueError(f"Unknown mode '{mode}', expected 'cache' or 'hf'")

    def row_to_content(self, row: dict) -> dict:
        # `question` is a plain string in every subset except `code`, where it
        # is a list of strings. Wrapping the list would nest it inside
        # item_content.input and fail the schema's list[str,dict] element check.
        question = row["question"]
        inputs = list(question) if isinstance(question, list) else [question]
        bad = [(i, type(x).__name__) for i, x in enumerate(inputs)
               if not isinstance(x, (str, dict))]
        if bad:
            raise ValueError(
                f"Row {row.get('question_id')} has non-str/dict input element(s) {bad}; "
                "item_content.input is list[str,dict]."
            )

        # Per-item taxonomy the schema has no dedicated field for. Bare values
        # for the columns `main` has always used (keeps previously converted
        # main items byte-identical); prefixed for the subset-specific ones,
        # whose values are meaningless without their column name. Everything is
        # str()-ed because benchmark_tags is list[str] and spatial's
        # `dimension` is an int.
        tags = [str(row[c]) for c in ("category", "task", "operation", "length")
                if row.get(c) is not None]
        if row.get("prompt") is not None:
            tags.append(f"prompt-style:{row['prompt']}")
        tags += [f"{c}:{row[c]}" for c in ("dimension", "mode")
                 if row.get(c) is not None]

        return {
            "input": inputs,
            # ground_truth is a single deterministic answer, stored as a string
            # in the source dataset (e.g. "[7, 30, 11, 54, 69]") -- kept as-is
            # rather than parsed, to stay faithful to the released data.
            "references": [row["ground_truth"]],
            "tags": tags,
        }

    def row_uid(self, row: dict) -> str:
        return row["question_id"]
