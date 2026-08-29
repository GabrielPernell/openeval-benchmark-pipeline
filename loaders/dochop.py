"""
loaders/dochop.py
==================
Adapter for DocHop ("Benchmarking Out-of-domain Multi-hop Reasoning in
Information-Dense Documents", OpenReview PQFkScoGqz).

STATUS (verified 2026-08-18): the dataset was made public by the authors and
is no longer blocked. It lives at HF `zhuoranyu336/dochop` as a single
2.4 GB TSV (`DocHop.tsv`, config `default`, split `train`, 2,074 rows). The
authors' evaluation code is at GitHub `ZhuoranYu/dochop-vlmevalkit`
(a VLMEvalKit fork; see `vlmeval/dataset/dochop.py`).

Columns:
    index        int   row position
    original_id  str   stable per-item id ("example-1", ...)
    image        str   base64-encoded PNG of a document page (text + charts),
                       ~755 KB per row -- NOT stored in the converted items,
                       see below
    question     str   the multi-hop question
    answer       str   the ground-truth answer (a single "pure value")
    task         str   ranking | fact_checking | value_retrieval | counting |
                       numeric_reasoning | hypothetical
    chart_num    int   number of charts on the page (2, 3, 4, 6)
    depth        int   reasoning depth (2-5)

Mapping:
    item_content.input      [<document image>, <question>]
    item_content.references [answer]
    benchmark_tags          task, depth:N, chart_num:N

About the image
---------------
Each page is a ~755 KB base64 blob; embedding all 2,074 would make a ~1.5 GB
items file. Two representations are supported, both typed dicts:
  * default -- a pointer to the source row:
      {"type": "document_image_ref",
       "content": {"dataset": ..., "split": ..., "row_index": N,
                   "column": "image", "encoding": "base64-png"}}
  * after `fetch_dochop_cache.py --images-dir ...` -- a relative path, the
    same convention loaders/visionwebdev.py uses for prototype images:
      {"type": "document_image", "content": "dochop_images/example-1.png"}

Scoring note (not used at conversion time, but relevant when responses are
added): the authors score with heuristic accuracy, extracting the text after
"The answer is:" and comparing numerically when both sides parse as numbers.
"""
import json

from adapter_base import DatasetAdapter

REPO_ID = "zhuoranyu336/dochop"
CONFIG = "default"
SPLIT = "train"


class DocHopAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__(
            benchmark_name="DocHop",
            # DocHop has a single public release with no subsets or version
            # tag, so the HF config name is the only stable identifier the
            # dataset itself provides. If the lab wants reproducibility against
            # a specific snapshot, pin the HF revision sha here instead.
            benchmark_version=CONFIG,
            paper_url="https://openreview.net/forum?id=PQFkScoGqz",
            dataset_url=f"https://huggingface.co/datasets/{REPO_ID}",
            base_tags=["multi-hop-reasoning", "document-understanding",
                       "chart-reasoning", "vision-language", "multimodal"],
        )

    def load_rows(self, source: str, limit: int | None = None, mode: str = "cache"):
        if mode == "cache":
            with open(source, encoding="utf-8") as f:
                rows = json.load(f)
            return rows[:limit] if limit else rows

        if mode == "hf":
            # Metadata-only read straight from the parquet conversion; the
            # image column is never transferred.
            from fetch_dochop_cache import fetch_rows
            return fetch_rows(limit=limit)

        raise ValueError(f"Unknown mode '{mode}', expected 'cache' or 'hf'")

    def row_to_content(self, row: dict) -> dict:
        question = str(row["question"])
        answer = row["answer"]
        if not question.strip():
            raise ValueError(f"Row {row.get('original_id')} has an empty question.")
        if answer is None or not str(answer).strip():
            raise ValueError(
                f"Row {row.get('original_id')} has an empty answer; item_content.references "
                "would be unscoreable."
            )

        # Image first, then the question -- the order the model receives them
        # in the authors' own prompt.
        if row.get("image_path"):
            image_entry = {"type": "document_image", "content": row["image_path"]}
        else:
            image_entry = {
                "type": "document_image_ref",
                "content": {
                    "dataset": REPO_ID,
                    "config": CONFIG,
                    "split": SPLIT,
                    "row_index": row["index"],
                    "column": "image",
                    "encoding": "base64-png",
                },
            }

        return {
            "input": [image_entry, question],
            # Answers are single "pure values" (an entity name or a number),
            # kept as released rather than coerced to a numeric type.
            "references": [str(answer)],
            "tags": [
                str(row["task"]),
                f"depth:{row['depth']}",
                f"chart_num:{row['chart_num']}",
            ],
        }

    def row_uid(self, row: dict) -> str:
        return str(row["original_id"])
