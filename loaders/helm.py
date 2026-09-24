"""
loaders/helm.py
================
Adapter for HELM run output, reading the cache built by
`fetchers/fetch_helm_cache.py`.

HELM is different from the other three benchmarks here in one important way:
its run folders contain the model responses as well as the items, so a HELM
import is a two-part job. This file handles the items; `add_helm_responses.py`
attaches the responses, the same split the upstream `helm_converter.py` uses
(`helm_to_items` / `helm_to_responses`).

One run folder is one model on one scenario, so the same item appears once per
model. Items are deduplicated by HELM's own instance id, and the loader checks
that runs agree about an item's content rather than trusting the first one it
reads.

Multiple choice
---------------
For `multiple_choice_joint` scenarios HELM stores the options as `references`,
with the right one tagged `correct`, and separately as an `output_mapping` from
answer letter to option text. Both were verified to be in the same order for
all 500 OpenBookQA instances, so the answer letter is the correct reference's
position: index 1 -> "B".

The repo layout follows the existing `mmlu_pro` items, which are the closest
precedent (also multiple choice, also HELM-derived responses): the question and
its options are one JSON string in `input[0]`, and `references` is one JSON
string of `{"answer": <letter>, "answer_index": <int>}`.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapter_base import DatasetAdapter  # noqa: E402

# HELM puts the paper behind a sentence like "Benchmark from <url>."
_URL_RE = re.compile(r"https?://[^\s)]+")


def paper_url_from(description: str) -> str:
    """The paper link HELM buries in a scenario's free-text description."""
    match = _URL_RE.search(description or "")
    return match.group(0).rstrip(".,;") if match else ""


def answer_letter(index: int) -> str:
    """multiple_choice_joint labels options A, B, C, ... in reference order."""
    return chr(ord("A") + index)


def read_manifest(cache_root: str) -> dict:
    path = os.path.join(cache_root, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no manifest.json in {cache_root}. Build the cache first:\n"
            f"    python fetchers/fetch_helm_cache.py --scenario commonsense"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class HELMAdapter(DatasetAdapter):
    """Items from one HELM scenario, pooled across every model run of it."""

    def __init__(self, subset: str = "commonsense"):
        # benchmark_name is refined in load_rows from the scenario's own name
        # (the run prefix is the scenario *group*: commonsense -> openbookqa).
        super().__init__(
            benchmark_name=subset,
            benchmark_version="",
            paper_url="",
            dataset_url="",
            base_tags=[],
        )
        self.subset = subset
        self.release = ""
        self.project = ""
        self.method = ""
        self.run_folders: list[str] = []

    @property
    def is_multiple_choice(self) -> bool:
        """Whether the answer is one of the references rather than free text.

        HELM's `multiple_choice_joint` puts the options in `references` and the
        model replies with a letter. Its `generation` methods put the expected
        answer there instead, and the model writes it out. The two need
        different reference mappings, and guessing wrong is silent: a
        generation item would get `{"answer": "A"}` for a question whose answer
        is a number.
        """
        return self.method.startswith("multiple_choice")

    # ---- loading -------------------------------------------------------
    def load_rows(self, source: str, limit: int | None = None, mode: str = "cache"):
        """Yield one row per distinct HELM instance across all runs of the scenario.

        `source` is the cache root written by fetch_helm_cache.py. `mode` is
        accepted for interface compatibility; HELM is always read from cache
        because a run folder is several MB of JSON per model.
        """
        if mode != "cache":
            raise ValueError(
                "HELM is only supported in --mode cache. Build the cache with "
                "fetchers/fetch_helm_cache.py, then pass its directory as --source."
            )

        manifest = read_manifest(source)
        self.project = manifest.get("project", "")
        self.release = manifest.get("release", "")
        runs = [e["folder"] for e in manifest["runs"]]
        if self.subset:
            runs = [f for f in runs if f.startswith(self.subset)]
        if not runs:
            raise ValueError(f"no runs for scenario {self.subset!r} in {source}")
        self.run_folders = runs

        by_id: dict[str, dict] = {}
        conflicts = 0
        for folder in runs:
            run_dir = os.path.join(source, folder)
            with open(os.path.join(run_dir, "scenario.json"), encoding="utf-8") as f:
                scenario = json.load(f)
            self._apply_scenario(scenario)

            with open(os.path.join(run_dir, "scenario_state.json"), encoding="utf-8") as f:
                state = json.load(f)
            method = state["adapter_spec"].get("method", "")
            if self.method and method != self.method:
                raise ValueError(
                    f"runs of {self.subset} disagree about the adaptation method "
                    f"({self.method!r} vs {method!r}); convert them separately.")
            self.method = method

            for req in state["request_states"]:
                inst = req["instance"]
                # Perturbed copies are variants of an item, not new items; the
                # responses side keeps them apart by perturbation.
                if "perturbation" in inst:
                    continue
                row = self._row(inst, req)
                seen = by_id.get(inst["id"])
                if seen is None:
                    by_id[inst["id"]] = row
                elif seen["question"] != row["question"] or seen["options"] != row["options"]:
                    # Two runs disagree about what this item says. Keep the
                    # first and count it rather than silently picking one.
                    conflicts += 1

        if conflicts:
            print(f"      ! {conflicts} instance(s) differ between runs; kept the first",
                  file=sys.stderr)

        rows = [by_id[k] for k in sorted(by_id)]
        print(f"      -> {len(rows)} distinct instances from {len(runs)} run(s) "
              f"({self.project} {self.release})", file=sys.stderr)
        return rows[:limit] if limit else rows

    def _apply_scenario(self, scenario: dict) -> None:
        """Fill the benchmark metadata from scenario.json (identical per run)."""
        self.benchmark_name = scenario.get("name") or self.subset
        # The release is the only thing that pins these items to a HELM run.
        # Existing HELM-derived rows in the repo leave benchmark_version empty,
        # so there is no way to tell which release they came from.
        self.benchmark_version = f"{self.project} {self.release}".strip()
        self.paper_url = paper_url_from(scenario.get("description", ""))
        self.dataset_url = scenario.get("definition_path", "")
        for tag in scenario.get("tags", []):
            if tag not in self.base_tags:
                self.base_tags.append(tag)

    @staticmethod
    def _row(inst: dict, req: dict) -> dict:
        refs = inst.get("references", [])
        options = [r["output"]["text"] for r in refs]
        correct = [i for i, r in enumerate(refs) if "correct" in (r.get("tags") or [])]
        return {
            "helm_instance_id": inst["id"],
            "split": inst.get("split", ""),
            "question": inst["input"]["text"],
            "options": options,
            "references_raw": refs,
            # A scenario with no tagged reference (generation tasks) leaves
            # these empty rather than guessing an answer.
            "answer_index": correct[0] if len(correct) == 1 else None,
            "answer": answer_letter(correct[0]) if len(correct) == 1 else "",
            "n_correct": len(correct),
        }

    # ---- mapping -------------------------------------------------------
    def row_to_content(self, row: dict) -> dict:
        """Nested JSON form: HELM's own reference dicts are kept intact.

        `tags` is only for tags a single row adds -- build_item concatenates it
        onto base_tags. HELM's tags describe the scenario, so they all live in
        base_tags and there is nothing per-item to add.
        """
        return {
            "input": [row["question"]],
            "references": row["references_raw"],
            "tags": [],
        }

    def row_uid(self, row: dict) -> str:
        return row["helm_instance_id"]

    # ---- HF repo layout ------------------------------------------------
    def repo_source_name(self) -> str:
        # Lowercase, so item_metadata.source actually matches the bench/ row.
        # 8 of the repo's 28 item tables currently have a source that resolves
        # to nothing, because they were written display-cased ('MMLU-Pro').
        return self.benchmark_name.lower()

    def repo_input(self, row: dict) -> dict:
        payload = {
            "question": row["question"],
            "split": row["split"],
            "helm_instance_id": row["helm_instance_id"],
        }
        # Only a multiple-choice item has options; for a generation scenario
        # `references` holds the expected answer, not a list to choose from.
        if self.is_multiple_choice:
            payload["options"] = row["options"]
        return payload

    def repo_references(self, row: dict) -> list:
        if not self.is_multiple_choice:
            # The expected answer(s) as written, matching how DocHop's
            # references carry the answer text.
            return [r["output"]["text"] for r in row["references_raw"]
                    if "correct" in (r.get("tags") or [])] or \
                   [r["output"]["text"] for r in row["references_raw"]]
        if row["answer_index"] is None:
            return []
        return [{"answer": row["answer"], "answer_index": row["answer_index"]}]
