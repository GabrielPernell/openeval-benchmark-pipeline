"""
loaders/visionwebdev.py
=========================
Adapter for "VisionWebDev" -- per the PhD student, the official benchmark name
to record in item_metadata.source.benchmark_name is "VisionWebDev", but the
released data lives under a different name, Vision2Web ("A Hierarchical
Benchmark for Visual Website Development with Agent Verification",
arXiv 2603.26648).

STATUS (verified 2026-08-16 against the real repos, not just the dataset card):
data is public and ungated at HF `zai-org/Vision2Web`; the evaluation harness
is at GitHub `zai-org/Vision2Web` (code only -- it contains no task data).

ACTUAL repo layout (the earlier dataset-card-based draft in this file was wrong
on nearly every point, so it has been replaced):

    HF repo root
      frontend/test.parquet     66 tasks   Level 2   prompt.txt  + prototypes
      webpage/test.parquet     100 tasks   Level 1   (no prompt) + prototypes
      website/test.parquet      27 tasks   Level 3   prd.md      + prototypes
      archives/{frontend,webpage,website}.tar.gz   the actual task directories
      -> 193 tasks total, matching the paper.

The three parquet files are a MANIFEST, not the data. Their columns are:
    task_name, level, workflow_steps, num_test_cases, resources_count,
    prototypes (list of image filenames),
    prompt_preview  (frontend only -- TRUNCATED with an ellipsis)
    prd_preview     (website only  -- TRUNCATED with an ellipsis)
The full prompt/PRD text, the prototype images, the resources, and above all
`workflow.json` (the ground truth) exist ONLY inside the tar.gz archives.

Per the harness (vision2web/core/dataset.py), a task directory is:
    <task_type>/<task_name>/
        prototypes/      UI mockup images (referenced by name in workflow.json)
        resources/       supporting assets (optional)
        prompt.txt       frontend tasks only
        prd.md           website tasks only
        workflow.json    ALWAYS present -- the ground truth

`workflow.json` is a list of workflow items (verified against real files):
    {
      "index": int,
      "depends_on": [int, ...],
      "summary": str,                       # what this workflow verifies
      "resolution": {"width": int, "height": int},
      "prototype": {"<name>": {"idx": int, "fullpage": bool}, ...},
      "content": [                          # functional test cases; [] for Level 1
        {"objective": str, "actions": [str], "validations": [str]}, ...
      ]
    }
Level 1 (webpage) tasks have `content: []` throughout -- they are pure visual
/screenshot workflows, which is why their manifest `num_test_cases` is 0.

Two modes:
  1. `mode="cache"`: read a local JSON file produced by
     fetch_vision2web_cache.py, which streams the tar.gz archives and keeps
     only the text (prompt/prd/workflow) -- complete items WITH references.
  2. `mode="hf"`: manifest-only fetch from the datasets-server REST API. Fast
     and tiny, but the rows carry no workflow.json, so items built this way
     have EMPTY references and cannot be scored. row_to_content() tags such
     items 'incomplete:manifest-only' so they can never be mistaken for
     finished conversions. Use this for indexing/inspection, not delivery.
"""
import json
import os

from adapter_base import DatasetAdapter
from net_utils import fetch_hf_rows

REPO_ID = "zai-org/Vision2Web"

# task_type -> (level, per-task prompt filename or None)
SUBSETS = {
    "webpage": ("Level 1", None),
    "frontend": ("Level 2", "prompt.txt"),
    "website": ("Level 3", "prd.md"),
}


class VisionWebDevAdapter(DatasetAdapter):
    def __init__(self, subset: str = "frontend"):
        if subset not in SUBSETS:
            raise ValueError(
                f"Unknown VisionWebDev subset '{subset}', expected one of {list(SUBSETS)}"
            )
        self.subset = subset
        super().__init__(
            benchmark_name="VisionWebDev",  # official name per PhD student, NOT "Vision2Web"
            # Like DSR-Bench, there's no semantic version; the meaningful
            # "version" is which of the three task types/difficulty levels
            # (webpage=L1, frontend=L2, website=L3) the item came from.
            benchmark_version=subset,
            paper_url="https://arxiv.org/abs/2603.26648",
            dataset_url=f"https://huggingface.co/datasets/{REPO_ID}",
            base_tags=["web-development", "code-generation", "vision-language",
                       "agent-verification", "multimodal"],
        )

    def load_rows(self, source: str, limit: int | None = None, mode: str = "cache"):
        if mode == "cache":
            with open(source, encoding="utf-8") as f:
                rows = json.load(f)
            # A cache file built for one task type shouldn't be silently
            # converted under another's benchmark_version.
            mismatched = {r.get("task_type") for r in rows} - {self.subset}
            if mismatched:
                raise ValueError(
                    f"--subset {self.subset} but cache {os.path.basename(source)} contains "
                    f"task_type(s) {sorted(mismatched)}. Re-run fetch_vision2web_cache.py for "
                    f"'{self.subset}', or pass the matching --subset."
                )
            return rows[:limit] if limit else rows

        if mode == "hf":
            rows = fetch_hf_rows(REPO_ID, config=self.subset, split="test", limit=limit)
            for r in rows:
                r["task_type"] = self.subset
                r["_manifest_only"] = True
            return rows

        raise ValueError(f"Unknown mode '{mode}', expected 'cache' or 'hf'")

    def row_to_content(self, row: dict) -> dict:
        level, _ = SUBSETS[self.subset]
        manifest_only = row.get("_manifest_only", False)

        # --- input: the task requirements text (if any) + the UI mockups ---
        # list[str|dict], so each prototype is a typed dict rather than a bare
        # filename, keeping the modality explicit for downstream consumers.
        inputs: list = []
        prompt_text = row.get("prompt") or row.get("prd")
        if manifest_only:
            # Only a truncated preview is available from the parquet manifest.
            prompt_text = row.get("prompt_preview") or row.get("prd_preview")
        if prompt_text:
            kind = "prd" if self.subset == "website" else "prompt"
            if manifest_only:
                kind += "_preview_truncated"
            inputs.append({"type": kind, "content": prompt_text})

        for proto in row.get("prototypes", []):
            inputs.append({
                "type": "prototype_image",
                # Path relative to the extracted archive root, so an item is
                # resolvable against a local checkout without absolute paths.
                "content": f"{self.subset}/{row['task_name']}/prototypes/{proto}",
            })

        # Level 1 tasks have no prompt at all -- the mockups ARE the spec. That
        # still satisfies item_content.input's non-empty rule via the images.
        if not inputs:
            raise ValueError(
                f"Task '{row.get('task_name')}' produced no input content "
                "(no prompt/prd and no prototypes) -- inspect the source row."
            )

        # --- references: the verification workflow = ground truth ---
        references: list = []
        workflow = row.get("workflow")
        if workflow is not None:
            references.append({"type": "verification_workflow", "content": workflow})

        # --- tags ---
        tags = [
            f"task-type:{self.subset}",
            f"level:{level.replace(' ', '').lower()}",
            f"workflows:{row.get('workflow_steps', len(workflow) if workflow else 0)}",
            f"test-cases:{row.get('num_test_cases', 0)}",
            f"prototypes:{len(row.get('prototypes', []))}",
        ]
        if manifest_only:
            # Deliberately loud: these items have empty references and are not
            # scoreable. See the module docstring.
            tags.append("incomplete:manifest-only")

        return {"input": inputs, "references": references, "tags": tags}

    def row_uid(self, row: dict) -> str:
        return f"{self.subset}/{row['task_name']}"
