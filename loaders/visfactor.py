"""
loaders/visfactor.py
====================
Adapter for VisFactor ("Human Cognitive Benchmarks Reveal Foundational Visual
Gaps in MLLMs", arXiv 2502.16435): 20 vision subtests from the
Factor-Referenced Cognitive Tests, digitised as image questions. Data at HF
`lmms-lab-encoder/visfactor` (one `test` split, 3,046 rows); evaluation code
at GitHub `CUHK-ARISE/VisFactor` (a VLMEvalKit fork,
vlmeval/dataset/visfactor.py). Fetch with fetchers/fetch_visfactor_cache.py,
which pins the HF revision.

Columns:
    index          int   row position, 0..3045
    category_name  str   subtest name ("Hidden Figures Test", ...)
    category_id    str   subtest code (CF1, ..., VZ3; 20 of them)
    eval_index     int   groups rows into 808 test questions per subtest; a
                         question counts as correct only if every row in its
                         group is (see visfactor_scoring.py)
    image          list  1-3 embedded images, referenced in `question` as
                         <IMAGE_0>, <IMAGE_1>, ...
    question       str   prompt template; `<br>` is a line break and
                         <ADDITIONAL_n> slots are filled from `additional`
    answer         str   T/F, a number, a letter, or "(r, c)" coordinates
    additional     str   ';'-separated fill-ins for the <ADDITIONAL_n> slots

Mapping, following the lab's DocHop re-upload (every source column kept under
its own name):
    item_content.input       [{every column except answer}]
    item_content.references  [{"answer": ...}]
    benchmark_tags           category_id

About the images
----------------
As in the DocHop re-upload, images are not embedded: `image` becomes a list of
{"url": <the row in the HF dataset viewer>, "image_index": k}, one per image,
where k is the N in <IMAGE_N>. Viewer row N is the row with index N in the
pinned file, which the fetcher checks.
"""
from adapter_base import DatasetAdapter

REPO_ID = "lmms-lab-encoder/visfactor"
VIEWER_ROW_URL = f"https://huggingface.co/datasets/{REPO_ID}/viewer/default/test?row={{}}"


def image_refs(row):
    url = VIEWER_ROW_URL.format(row["index"])
    return [{"url": url, "image_index": k} for k in range(row["n_images"])]


class VisFactorAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__(
            benchmark_name="VisFactor",
            # The VLMEvalKit dataset key; VisFactor_GE/_GN/_GH are the
            # generated easy/normal/hard variants, not converted here.
            benchmark_version="VisFactor",
            paper_url="https://arxiv.org/abs/2502.16435",
            dataset_url=f"https://huggingface.co/datasets/{REPO_ID}",
            base_tags=["visual-cognition", "psychometrics", "vision-language", "multimodal"],
        )

    def load_rows(self, source: str, limit: int | None = None, mode: str = "cache"):
        import os as _os, sys as _sys
        _fetchers = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "fetchers")
        if _fetchers not in _sys.path:
            _sys.path.insert(0, _fetchers)
        import fetch_visfactor_cache

        if mode == "hf":
            source = fetch_visfactor_cache.fetch_items(
                _os.path.dirname(source) if source else "data_cache/visfactor")
        elif mode != "cache":
            raise ValueError(f"Unknown mode '{mode}', expected 'cache' or 'hf'")

        rows = fetch_visfactor_cache.load_items(source)
        return rows[:limit] if limit else rows

    def _input(self, row):
        if not row["question"].strip():
            raise ValueError(f"VisFactor row {row['index']} has an empty question")
        # Count image tags in the prompt as the authors build it: some (all of
        # subtest I3) arrive through an <ADDITIONAL_n> slot.
        from visfactor_scoring import build_prompt
        shown = {k for kind, k in build_prompt(row["question"], row["additional"], row["n_images"])
                 if kind == "image"}
        if shown != set(range(row["n_images"])):
            raise ValueError(f"VisFactor row {row['index']}: {row['n_images']} images, "
                             f"prompt shows {sorted(shown)}")
        return {
            "index": row["index"],
            "category_name": row["category_name"],
            "category_id": row["category_id"],
            "eval_index": row["eval_index"],
            "image": image_refs(row),
            "question": row["question"],
            "additional": row["additional"],
        }

    def _references(self, row):
        if not str(row["answer"]).strip():
            raise ValueError(f"VisFactor row {row['index']} has an empty answer")
        return {"answer": row["answer"]}

    def row_to_content(self, row: dict) -> dict:
        return {
            "input": [self._input(row)],
            "references": [self._references(row)],
            "tags": [row["category_id"]],
        }

    def repo_input(self, row: dict) -> dict:
        return self._input(row)

    def repo_references(self, row: dict) -> list:
        return [self._references(row)]

    def row_uid(self, row: dict) -> str:
        return str(row["index"])
